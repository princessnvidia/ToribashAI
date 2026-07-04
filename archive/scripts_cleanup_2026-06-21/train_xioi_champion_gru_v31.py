#!/usr/bin/env python3
"""
train_xioi_champion_gru_v31.py

Overfit GRU sur le champion V30 promu.
Sortie: models/xioi_champion_gru_v31.pt
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET_PATH = ROOT / "datasets" / "ml" / "xioi_champion_v31_sequences.jsonl"
MODEL_PATH = ROOT / "models" / "xioi_champion_gru_v31.pt"
SUMMARY_PATH = ROOT / "generated_replays" / "xioi_champion_gru_v31_summary.json"

SEED = 3101
BATCH_SIZE = 32
EPOCHS = 450
LR = 8e-4
HIDDEN = 224
NUM_LAYERS = 2
DROPOUT = 0.0
ACTION_DIM = 20
CLASSES = 5

random.seed(SEED)
torch.manual_seed(SEED)


class SeqDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        return torch.tensor(r["state_seq"], dtype=torch.float32), torch.tensor(r["action"], dtype=torch.long)


class ChampionGRU(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.gru = nn.GRU(state_dim, HIDDEN, num_layers=NUM_LAYERS, batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, ACTION_DIM * CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        z = out[:, -1]
        return self.head(z).view(-1, ACTION_DIM, CLASSES)


def load_rows() -> list[dict[str, Any]]:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset absent: {DATASET_PATH}. Lance build_xioi_champion_dataset_v31.py")
    rows = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("Dataset vide")
    return rows


def weights_for(rows: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    counts = torch.ones(CLASSES, dtype=torch.float32)
    for r in rows:
        for v in r["action"]:
            counts[int(v)] += 1.0
    inv = counts.sum() / counts
    w = inv / inv.mean()
    # Le champion contient beaucoup de 0; on baisse 0 mais pas autant que V27.
    w[0] *= 0.36
    w = w / w.mean()
    return w.to(device)


def evaluate(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, criterion) -> dict[str, Any]:
    loader = DataLoader(SeqDataset(rows), batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    total_loss = 0.0
    total = correct = exact = 0
    nonzero_total = nonzero_correct = 0
    true_counts: Counter[int] = Counter()
    pred_counts: Counter[int] = Counter()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits.reshape(-1, CLASSES), y.reshape(-1))
            pred = logits.argmax(dim=-1)
            total_loss += float(loss.item()) * x.size(0)
            total += y.numel()
            correct += int((pred == y).sum().item())
            exact += int((pred == y).all(dim=1).sum().item())
            mask = y != 0
            nonzero_total += int(mask.sum().item())
            nonzero_correct += int(((pred == y) & mask).sum().item())
            for v in y.reshape(-1).cpu().tolist(): true_counts[int(v)] += 1
            for v in pred.reshape(-1).cpu().tolist(): pred_counts[int(v)] += 1
    return {
        "loss": total_loss / max(1, len(rows)),
        "joint_acc": correct / max(1, total),
        "exact_acc": exact / max(1, len(rows)),
        "nonzero_acc": nonzero_correct / max(1, nonzero_total),
        "true_counts": dict(true_counts),
        "pred_counts": dict(pred_counts),
    }


def main() -> None:
    rows = load_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dim = len(rows[0]["state_seq"][0])
    seq_len = len(rows[0]["state_seq"])
    print("Device:", device)
    print("Dataset:", len(rows), "state_dim", state_dim, "seq_len", seq_len)
    print("Source:", rows[0].get("source"))

    loader = DataLoader(SeqDataset(rows), batch_size=BATCH_SIZE, shuffle=True)
    weights = weights_for(rows, device)
    print("Class weights:", [round(float(x), 4) for x in weights.detach().cpu()])

    model = ChampionGRU(state_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_metric = -1.0
    best = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total = correct = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits.reshape(-1, CLASSES), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            pred = logits.argmax(dim=-1)
            total_loss += float(loss.item()) * x.size(0)
            total += y.numel()
            correct += int((pred == y).sum().item())
        scheduler.step()

        if epoch == 1 or epoch % 10 == 0 or epoch >= EPOCHS - 5:
            metrics = evaluate(model, rows, device, criterion)
            metric = metrics["joint_acc"] + metrics["nonzero_acc"] + 2.0 * metrics["exact_acc"]
            print(
                f"Epoch {epoch:03d} | train_loss={total_loss/max(1,len(rows)):.4f} "
                f"acc={correct/max(1,total):.4f} | all_loss={metrics['loss']:.4f} "
                f"joint={metrics['joint_acc']:.4f} exact={metrics['exact_acc']:.4f} nonzero={metrics['nonzero_acc']:.4f}"
            )
            if metric > best_metric:
                best_metric = metric
                best = metrics
                torch.save({
                    "version": 31,
                    "mode": "xioi_champion_overfit",
                    "model_state": model.state_dict(),
                    "state_dim": state_dim,
                    "seq_len": seq_len,
                    "hidden": HIDDEN,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "action_dim": ACTION_DIM,
                    "classes": CLASSES,
                    "dataset": str(DATASET_PATH),
                    "source": rows[0].get("source"),
                    "metrics": metrics,
                }, MODEL_PATH)
                print("  saved", MODEL_PATH)

    SUMMARY_PATH.write_text(json.dumps({
        "version": 31,
        "dataset": str(DATASET_PATH),
        "model": str(MODEL_PATH),
        "best_metric": best_metric,
        "best_metrics": best,
    }, indent=2), encoding="utf-8")
    print("Done. Best metric:", best_metric)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
