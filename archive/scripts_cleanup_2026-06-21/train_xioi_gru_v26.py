#!/usr/bin/env python3
"""
train_xioi_gru_v26.py

GRU Xioi-only, entraîné sur la boucle/début de marche Xioi V26.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET_PATH = ROOT / "datasets" / "ml" / "xioi_only_v26_sequences.jsonl"
MODEL_PATH = ROOT / "models" / "xioi_gru_v26.pt"

SEED = 26
BATCH_SIZE = 64
EPOCHS = 120
LR = 1e-3
HIDDEN = 128
NUM_LAYERS = 1
DROPOUT = 0.0

random.seed(SEED)
torch.manual_seed(SEED)


class SeqDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        x = torch.tensor(r["state_seq"], dtype=torch.float32)
        y = torch.tensor(r["action"], dtype=torch.long)
        return x, y


class XioiGRU(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=HIDDEN,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN),
            nn.Linear(HIDDEN, 20 * 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        z = out[:, -1, :]
        return self.head(z).view(-1, 20, 5)


def load_rows() -> list[dict[str, Any]]:
    rows = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("Dataset vide")
    return rows


def split_rows(rows: list[dict[str, Any]]):
    # Split temporel: train début/milieu, val fin. Plus honnête pour une seule séquence.
    n = len(rows)
    cut = max(1, int(n * 0.82))
    return rows[:cut], rows[cut:]


def class_weights(rows: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    counts = torch.ones(5, dtype=torch.float32)
    for r in rows:
        for v in r["action"]:
            counts[int(v)] += 1.0
    inv = counts.sum() / counts
    weights = inv / inv.mean()
    # On évite d'écraser totalement la classe 0, mais on booste les actions.
    weights[0] *= 0.55
    weights = weights / weights.mean()
    return weights.to(device)


def eval_model(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    exact = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, 5), y.reshape(-1))
            pred = logits.argmax(dim=-1)
            total_loss += float(loss.item()) * x.size(0)
            total += y.numel()
            correct += int((pred == y).sum().item())
            exact += int((pred == y).all(dim=1).sum().item())
    nseq = len(loader.dataset)
    return total_loss / max(1, nseq), correct / max(1, total), exact / max(1, nseq)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows()
    train_rows, val_rows = split_rows(rows)
    state_dim = len(rows[0]["state_seq"][0])

    print("Device:", device)
    print("Dataset:", len(rows), "train", len(train_rows), "val", len(val_rows), "state_dim", state_dim)

    train_loader = DataLoader(SeqDataset(train_rows), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SeqDataset(val_rows), batch_size=BATCH_SIZE, shuffle=False)

    weights = class_weights(train_rows, device)
    print("Class weights:", [float(x) for x in weights.cpu()])

    model = XioiGRU(state_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best = float("inf")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, 5), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            pred = logits.argmax(dim=-1)
            total_loss += float(loss.item()) * x.size(0)
            total += y.numel()
            correct += int((pred == y).sum().item())

        train_loss = total_loss / max(1, len(train_rows))
        train_acc = correct / max(1, total)
        val_loss, val_acc, val_exact = eval_model(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_acc:.4f} exact={val_exact:.4f}"
        )

        if val_loss < best:
            best = val_loss
            torch.save({
                "version": 26,
                "model_state": model.state_dict(),
                "state_dim": state_dim,
                "hidden": HIDDEN,
                "num_layers": NUM_LAYERS,
                "seq_len": len(rows[0]["state_seq"]),
                "action_dim": 20,
                "classes": 5,
                "dataset": str(DATASET_PATH),
                "best_val_loss": best,
            }, MODEL_PATH)
            print("  saved", MODEL_PATH)

    print("Done. Best:", best)


if __name__ == "__main__":
    main()
