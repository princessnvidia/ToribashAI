#!/usr/bin/env python3
"""
train_curated_walking_gru_v23.py

GRU marche curaté V23 : entraîné uniquement sur les débuts de marche repérés par Vio.
Sortie : models/curated_walking_gru_v23.pt
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET = ROOT / "datasets" / "ml" / "curated_walking_v23_sequences.jsonl"
MODEL_OUT = ROOT / "models" / "curated_walking_gru_v23.pt"

EPOCHS = 20
BATCH_SIZE = 128
LR = 5e-4
HIDDEN = 128
LAYERS = 1
DROPOUT = 0.15
VAL_RATIO = 0.15
SEED = 23


class WalkingDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.rows.append(json.loads(line))
        if not self.rows:
            raise RuntimeError(f"Dataset vide: {path}")
        self.state_dim = len(self.rows[0]["state_seq"][0])
        self.seq_len = len(self.rows[0]["state_seq"])
        self.action_dim = len(self.rows[0]["action"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        x = torch.tensor(r["state_seq"], dtype=torch.float32)
        y = torch.tensor(r["action"], dtype=torch.long)
        return x, y


class WalkingGRU(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 20):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=HIDDEN,
            num_layers=LAYERS,
            batch_first=True,
            dropout=0.0 if LAYERS == 1 else DROPOUT,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, action_dim * 5),
        )
        self.action_dim = action_dim

    def forward(self, x):
        out, _ = self.gru(x)
        h = out[:, -1, :]
        logits = self.head(h)
        return logits.view(-1, self.action_dim, 5)


def class_weights(ds: WalkingDataset) -> torch.Tensor:
    counts = torch.ones(5, dtype=torch.float32)
    for r in ds.rows:
        for v in r["action"]:
            counts[int(v)] += 1
    inv = counts.sum() / counts
    # soft weights, éviter que les classes rares explosent trop.
    w = torch.sqrt(inv)
    w = w / w.mean()
    return w


def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_joint = 0
    correct_joint = 0
    exact = 0
    rows = 0
    pred_counts = torch.zeros(5, dtype=torch.long)
    true_counts = torch.zeros(5, dtype=torch.long)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, 5), y.reshape(-1))
            total_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=-1)
            correct_joint += int((pred == y).sum().item())
            total_joint += y.numel()
            exact += int((pred == y).all(dim=1).sum().item())
            rows += x.size(0)
            for i in range(5):
                pred_counts[i] += int((pred == i).sum().item())
                true_counts[i] += int((y == i).sum().item())
    return {
        "loss": total_loss / max(1, rows),
        "joint_acc": correct_joint / max(1, total_joint),
        "exact": exact / max(1, rows),
        "pred_counts": pred_counts.tolist(),
        "true_counts": true_counts.tolist(),
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    ds = WalkingDataset(DATASET)
    print("Dataset:", len(ds), "seq_len:", ds.seq_len, "state_dim:", ds.state_dim, "action_dim:", ds.action_dim)

    val_n = max(1, int(len(ds) * VAL_RATIO))
    train_n = len(ds) - val_n
    train_ds, val_ds = random_split(ds, [train_n, val_n], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = WalkingGRU(ds.state_dim, ds.action_dim).to(device)
    weights = class_weights(ds).to(device)
    print("Class weights:", weights.detach().cpu().tolist())
    criterion = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best = None
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total = 0.0
        seen = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, 5), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * x.size(0)
            seen += x.size(0)
        val = eval_model(model, val_loader, criterion, device)
        train_loss = total / max(1, seen)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val['loss']:.4f} | val_joint={val['joint_acc']:.4f} | exact={val['exact']:.4f}")
        if best is None or val["loss"] < best["loss"]:
            best = {"epoch": epoch, **val}
            torch.save({
                "model_state": model.state_dict(),
                "state_dim": ds.state_dim,
                "seq_len": ds.seq_len,
                "action_dim": ds.action_dim,
                "hidden": HIDDEN,
                "layers": LAYERS,
                "dropout": DROPOUT,
                "best": best,
                "dataset": str(DATASET),
            }, MODEL_OUT)
            print("  saved:", MODEL_OUT)

    print("Best:", best)


if __name__ == "__main__":
    main()
