import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding

def extract_label_name(ex: dict) -> str:
    labels = ex.get("labels", {})
    if "clause-type" in labels:
        return labels["clause-type"]
    if "label" in labels:
        return labels["label"]
    if isinstance(labels, dict) and len(labels) > 0:
        return next(iter(labels.values()))
    raise KeyError(f"labels not found or empty in example: {ex.keys()}")

def get_clause_text(ex: dict) -> str:
    if "text" in ex:
        return (ex["text"] or "").strip()
    msgs = ex.get("messages", [])
    for m in msgs:
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return "\n".join((m.get("content") or "") for m in msgs).strip()

def build_input(tokenizer, text: str) -> dict:
    return tokenizer(text, truncation=True)

def load_jsonl_dataset(path: str, tokenizer, label_list: List[str]):
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.select(range(0, 1000))

    label2id = {name: i for i, name in enumerate(label_list)}
    id2label = {i: n for n, i in label2id.items()}

    def map_fn(ex):
        text = get_clause_text(ex)
        lab_name = extract_label_name(ex)
        enc = build_input(tokenizer, text)
        enc["labels"] = label2id[lab_name]
        return enc

    cols = {"input_ids", "attention_mask", "labels"}
    keep_cols = [c for c in cols if c in ds.column_names]
    ds = ds.map(
        map_fn,
        remove_columns=[c for c in ds.column_names if c not in keep_cols and c not in ("text", "messages", "labels")],
        num_proc=min(8, os.cpu_count() or 1),
        desc="Tokenizing",
    )
    ds = ds.with_format("torch", columns=list(cols & set(ds.column_names)))
    return ds, label_list, label2id, id2label

class MeanPoolerHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = summed / denom
        pooled = self.dropout(pooled)
        pooled = pooled.to(self.classifier.weight.dtype)
        logits = self.classifier(pooled)
        return logits

class CausalLMWithClsHead(nn.Module):
    def __init__(self, base_model, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.base = base_model
        self.backbone = self._resolve_backbone(self.base)
        hidden = self.backbone.config.hidden_size

        base_dtype = next(self.base.parameters()).dtype
        self.head = MeanPoolerHead(hidden, num_labels, dropout).to(dtype=base_dtype)

    @staticmethod
    def _resolve_backbone(m):
        """
        Return the decoder/backbone that outputs last_hidden_state.
        Works for:
          - plain *ForCausalLM (has .model)
          - PEFT-wrapped models (has .get_base_model().model)
        """
        if hasattr(m, "get_base_model"):
            m = m.get_base_model()
        if hasattr(m, "model"):
            return m.model
        if hasattr(m, "base_model") and hasattr(m.base_model, "model"):
            return m.base_model.model
        raise AttributeError("Could not locate decoder backbone with last_hidden_state")

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        last_h = outputs.last_hidden_state  # (B,T,H)
        del outputs

        if next(self.head.parameters()).device != last_h.device:
            self.head.to(last_h.device)

        attention_mask = attention_mask.to(last_h.device)
        logits = self.head(last_h, attention_mask)

        out = {"logits": logits}
        if labels is not None:
            out["loss"] = nn.CrossEntropyLoss()(logits.float(), labels.to(logits.device))
        return out


@torch.no_grad()
def infer(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in dataloader:
        labels = batch["labels"].tolist()
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch)["logits"]
        preds = logits.argmax(dim=-1).tolist()
        y_true.extend(labels)
        y_pred.extend(preds)
    return y_true, y_pred

def maybe_enable_lora(base, args, state_dict_keys):
    need_lora = args.use_lora or any("lora_" in k for k in state_dict_keys)
    if not need_lora:
        return base
    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise RuntimeError("LoRA weights detected/requested but `peft` is not installed.") from e

    print("→ Enabling LoRA adapters before loading state dict...")
    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    base = get_peft_model(base, lcfg)
    target_dtype = torch.bfloat16 if args.bf16 else torch.float16
    base = base.to(dtype=target_dtype)
    return base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default='./data/jsonl/ledgar_test_text.jsonl', help="Path to JSONL to evaluate (test/val).")
    ap.add_argument("--ckpt_dir", default='./outputs_ministral8b_headlora2', help="Training output directory (contains tokenizer + label2id.json + pytorch_model.bin).")
    ap.add_argument("--base_model", default="mistralai/Ministral-8B-Instruct-2410")
    ap.add_argument("--out_json", default="./data/evaluations/ministral8B_headlora.json")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--bf16", action="store_true", default=False)
    ap.add_argument("--max_len", type=int, default=2048)
    ap.add_argument("--use_lora", action="store_true", default=False)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    ckpt = Path(args.ckpt_dir)
    assert ckpt.exists(), f"Checkpoint dir not found: {ckpt}"

    tokenizer = AutoTokenizer.from_pretrained(ckpt, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_len

    with open(ckpt / "label2id.json") as f:
        label2id: Dict[str, int] = json.load(f)
    label_list: List[str] = [None] * len(label2id)
    for name, i in label2id.items():
        label_list[i] = name
    id2label = {i: n for i, n in enumerate(label_list)}

    ds, _, _, _ = load_jsonl_dataset(args.jsonl, tokenizer, label_list)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    print("Loading base...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto",
    )
    base.config.use_cache = False

    model = CausalLMWithClsHead(base, num_labels=len(label_list), dropout=0.1)

    sd_path = ckpt / "pytorch_model.bin"
    assert sd_path.exists(), f"Missing weights: {sd_path}"
    state_dict = torch.load(sd_path, map_location="cpu")
    model.base = maybe_enable_lora(model.base, args, state_dict.keys())

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"⚠️ Missing keys when loading: {len(missing)} (likely fine if base was frozen).")
    if unexpected:
        print(f"⚠️ Unexpected keys when loading: {len(unexpected)} (ensure --use_lora if you trained with LoRA).")

    device = next(model.parameters()).device

    y_true, y_pred = infer(model, dl, device)

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"y_true": y_true, "y_pred": y_pred}, f, indent=2)
    print(f"\n Saved predictions to {out_path}")

    y_true_n = [id2label[i] for i in y_true]
    y_pred_n = [id2label[i] for i in y_pred]
    print("\nClassification report (by label name):")
    print(classification_report(y_true_n, y_pred_n, digits=3))

if __name__ == "__main__":
    main()
