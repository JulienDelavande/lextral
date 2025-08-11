import os
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Optional

from sklearn import base
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import load_dataset
from sklearn.metrics import classification_report

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model

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
    """Retourne le texte de la clause depuis 'text' si présent,
    sinon extrait le 1er message user depuis 'messages'."""
    if "text" in ex:
        return (ex["text"] or "").strip()

    msgs = ex.get("messages", [])
    for m in msgs:
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return "\n".join((m.get("content") or "") for m in msgs).strip()

def build_input(tokenizer, text: str) -> dict:
    return tokenizer(text, truncation=True)

def load_jsonl_dataset(path: str, tokenizer, label_list: list | None = None):
    from datasets import load_dataset
    ds = load_dataset("json", data_files=path, split="train")

    if label_list is None:
        all_labels = sorted({extract_label_name(ex) for ex in ds})
        label_list = all_labels
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
        mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)  # (B,T,1)
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

def train_one_epoch(model, dataloader, optimizer, scheduler, device, grad_accum=1, max_norm=1.0, log_every=50):
    model.train()
    total_loss = 0.0
    step = 0
    optimizer.zero_grad()
    for i, batch in enumerate(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out["loss"] / grad_accum
        loss.backward()
        total_loss += loss.item()
        if (i + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % log_every == 0:
                print(f"step {step} | loss {total_loss / log_every:.4f}")
                total_loss = 0.0


@torch.no_grad()
def evaluate(model, dataloader, id2label, device):
    model.eval()
    y_true, y_pred = [], []
    for batch in dataloader:
        labels = batch["labels"].tolist()
        batch = {k: v.to(device) for k, v in batch.items()}
        logits = model(**batch)["logits"]
        preds = logits.argmax(dim=-1).tolist()
        y_true.extend(labels)
        y_pred.extend(preds)
    y_true_n = [id2label[i] for i in y_true]
    y_pred_n = [id2label[i] for i in y_pred]
    print(classification_report(y_true_n, y_pred_n, digits=3))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_jsonl", default="./data/jsonl/ledgar_train_text.jsonl")
    p.add_argument("--val_jsonl", default="./data/jsonl/ledgar_validation_text.jsonl")
    p.add_argument("--base_model", default="mistralai/Ministral-8B-Instruct-2410")
    p.add_argument("--output_dir", default="./outputs_ministral8b_headlora")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--lora", action="store_true", help="Also LoRA-tune the base (optional)")
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_len

    print("Loading base...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        device_map="auto",
    )
    base.config.use_cache = False  # nécessaire en train

    if args.lora:
        print("Enabling LoRA on base...")
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
    else:
        for p_ in base.parameters():
            p_.requires_grad = False


    print("Preparing data...")
    train_ds, label_list, label2id, id2label = load_jsonl_dataset(args.train_jsonl, tokenizer)
    val_ds = None
    if args.val_jsonl:
        val_ds, _, _, _ = load_jsonl_dataset(args.val_jsonl, tokenizer, label_list)

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator) if val_ds else None

    model = CausalLMWithClsHead(base, num_labels=len(label_list), dropout=args.dropout)
    device = next(model.parameters()).device

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    total_steps = math.ceil(len(train_dl) / args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    print(f"Training: epochs={args.epochs}, steps={total_steps}, labels={len(label_list)}")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_one_epoch(model, train_dl, optimizer, scheduler, device,
                        grad_accum=args.grad_accum, log_every=50)
        if val_dl:
            print("\nValidation:")
            evaluate(model, val_dl, id2label, device)

    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    with open(os.path.join(args.output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved to {args.output_dir}")
    if args.lora:
        print("You also trained LoRA adapters inside the base; saving full state dict includes them.")
    else:
        print("Only the classification head (and not the base) was trained.")
    

if __name__ == "__main__":
    main()
