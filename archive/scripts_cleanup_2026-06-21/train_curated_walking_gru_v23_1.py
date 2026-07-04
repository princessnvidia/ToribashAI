#!/usr/bin/env python3
"""
train_curated_walking_gru_v23_1.py

Petit GRU action imitation pour le dataset curated_walking_v23_1.
À lancer seulement après validation visuelle des RPL exportés.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET = ROOT / "datasets" / "ml" / "curated_walking_v23_1_sequences.jsonl"
MODEL_OUT = ROOT / "models" / "curated_walking_gru_v23_1.pt"

BATCH_SIZE = 64
EPOCHS = 40
LR = 5e-4
HIDDEN = 128
LAYERS = 1
SEED = 23

class WalkDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    self.rows.append((r["state_seq"], r["action"]))
        if not self.rows:
            raise RuntimeError("Dataset vide")
        self.seq_len = len(self.rows[0][0])
        self.state_dim = len(self.rows[0][0][0])

    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        x, y = self.rows[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

class GRUAction(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.gru = nn.GRU(state_dim, HIDDEN, LAYERS, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, 20 * 5))
    def forward(self, x):
        out, _ = self.gru(x)
        z = out[:, -1]
        return self.head(z).view(-1, 20, 5)


def main():
    random.seed(SEED); torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = WalkDataset(DATASET)
    n_val = max(1, int(len(ds) * 0.15))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    counts = Counter()
    for _, y in ds:
        counts.update(y.tolist())
    weights = torch.ones(5)
    total = sum(counts.values())
    for k in range(5):
        weights[k] = (total / max(1, counts[k])) ** 0.25
    weights = (weights / weights.mean()).to(device)

    model = GRUAction(ds.state_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best = 1e9
    print("Device:", device)
    print("Dataset:", len(ds), "train", n_train, "val", n_val, "state_dim", ds.state_dim)
    print("Class counts:", dict(counts))
    print("Class weights:", weights.detach().cpu().tolist())

    for ep in range(1, EPOCHS + 1):
        model.train(); tr_loss = 0; tr_acc = 0; tr_total = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, 5), y.reshape(-1))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tr_loss += loss.item() * x.size(0)
            pred = logits.argmax(-1)
            tr_acc += (pred == y).sum().item(); tr_total += y.numel()
        model.eval(); va_loss = 0; va_acc = 0; va_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = loss_fn(logits.reshape(-1, 5), y.reshape(-1))
                va_loss += loss.item() * x.size(0)
                pred = logits.argmax(-1)
                va_acc += (pred == y).sum().item(); va_total += y.numel()
        tr_loss /= max(1, n_train); va_loss /= max(1, n_val)
        print(f"Epoch {ep:02d} | train_loss={tr_loss:.4f} acc={tr_acc/max(1,tr_total):.4f} | val_loss={va_loss:.4f} acc={va_acc/max(1,va_total):.4f}")
        if va_loss < best:
            best = va_loss
            MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "version": "23.1",
                "model_state": model.state_dict(),
                "state_dim": ds.state_dim,
                "seq_len": ds.seq_len,
                "hidden": HIDDEN,
                "layers": LAYERS,
                "best_val_loss": best,
            }, MODEL_OUT)
            print("  saved", MODEL_OUT)

if __name__ == "__main__":
    main()
