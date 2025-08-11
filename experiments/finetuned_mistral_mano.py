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

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model


SYSTEM_PROMPT = (
    "You are a contract clause classifier. "
    "Classify the user's clause into one of the known categories. "
    "Respond with only the category name."
)


def extract_label_name(ex: Dict) -> str:
    # adapte si ta clé diffère
    labels = ex.get("labels", {})
    return labels.get("clause-type") or labels.get("label") or list(labels.values())[0]


def first_user_text(messages):
    # Si ton JSONL contient déjà la clause brute, prends-la directement.
    # Sinon, récupère le premier message 'user'.
    for m in messages:
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return "\n".join((m.get("content") or "") for m in messages).strip()

def build_input(tokenizer, text: str) -> dict:
    # PLUS DE CHAT TEMPLATE : on tokenize la clause telle quelle
    return tokenizer(text, truncation=True)


def load_jsonl_dataset(path: str, tokenizer, label_list: Optional[List[str]] = None):
    ds = load_dataset("json", data_files=path, split="train")

    if label_list is None:
        all_labels = sorted({extract_label_name(ex) for ex in ds})
        label_list = all_labels
    label2id = {name: i for i, name in enumerate(label_list)}
    id2label = {i: n for n, i in label2id.items()}

    def map_fn(ex):
        text = first_user_text(ex["messages"])
        lab = extract_label_name(ex)
        enc = build_input(tokenizer, text)
        enc["labels"] = label2id[lab]
        return enc

    cols = ["input_ids", "attention_mask", "labels"]
    ds = ds.map(map_fn, remove_columns=[c for c in ds.column_names if c not in cols])
    ds = ds.with_format("torch", columns=cols)
    return ds, label_list, label2id, id2label


class MeanPoolerHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, last_hidden_state, attention_mask):
        # mean pooling masquée (B, T, H) -> (B, H)
        mask = attention_mask.unsqueeze(-1)  # (B,T,1)
        summed = (last_hidden_state * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        pooled = summed / denom
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


class CausalLMWithClsHead(nn.Module):
    def __init__(self, base_model, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.base = base_model
        hidden = base_model.config.hidden_size
        self.head = MeanPoolerHead(hidden, num_labels, dropout)

    def forward(self, input_ids, attention_mask, labels=None):
        # IMPORTANT: use_cache False en train
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=False,
        )
        last_h = outputs.last_hidden_state  # (B,T,H)
        logits = self.head(last_h, attention_mask)
        out = {"logits": logits}
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
            out["loss"] = loss
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
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--val_jsonl", default=None)
    p.add_argument("--base_model", default="mistralai/Ministral-3B-Instruct")
    p.add_argument("--output_dir", default="./outputs_ministral3b_head")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--lora", action="store_true", help="Also LoRA-tune the base (optional)")
    p.add_argument("--lora_r", type=int, default=8)
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
    else:
        # gèle entièrement la base si on ne veut entraîner que la tête
        for p_ in base.parameters():
            p_.requires_grad = False

    # datasets
    print("Preparing data...")
    train_ds, label_list, label2id, id2label = load_jsonl_dataset(args.train_jsonl, tokenizer)
    val_ds = None
    if args.val_jsonl:
        val_ds, _, _, _ = load_jsonl_dataset(args.val_jsonl, tokenizer, label_list)

    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator) if val_ds else None

    # model with head
    model = CausalLMWithClsHead(base, num_labels=len(label_list), dropout=args.dropout)
    device = next(model.parameters()).device

    # optimiser & scheduler
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    total_steps = math.ceil(len(train_dl) / args.grad_accum) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # loop
    print(f"Training: epochs={args.epochs}, steps={total_steps}, labels={len(label_list)}")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_one_epoch(model, train_dl, optimizer, scheduler, device,
                        grad_accum=args.grad_accum, log_every=50)
        if val_dl:
            print("\nValidation:")
            evaluate(model, val_dl, id2label, device)

    # save head + label mapping
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    with open(os.path.join(args.output_dir, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=2)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Saved to {args.output_dir}")
    if args.lora:
        print("ℹ️ You also trained LoRA adapters inside the base; saving full state dict includes them.")
    else:
        print("ℹ️ Only the classification head (and not the base) was trained.")
    

if __name__ == "__main__":
    main()
