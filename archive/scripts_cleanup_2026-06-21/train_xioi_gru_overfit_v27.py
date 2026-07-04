#!/usr/bin/env python3
"""
train_xioi_gru_overfit_v27.py

Étape 1 / V27:
  Overfit volontaire sur le dataset Xioi-only V26.

But:
  - ne pas chercher la généralisation
  - vérifier que l'architecture peut mémoriser la mécanique Xioi
  - sauvegarder un modèle "teacher" pour génération guidée

Sortie:
  models/xioi_gru_overfit_v27.pt
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
DATASET_PATH = ROOT / "datasets" / "ml" / "xioi_only_v26_sequences.jsonl"
MODEL_PATH = ROOT / "models" / "xioi_gru_overfit_v27.pt"
SUMMARY_PATH = ROOT / "generated_replays" / "xioi_gru_overfit_v27_summary.json"

SEED = 2701
BATCH_SIZE = 32
EPOCHS = 500
LR = 8e-4
HIDDEN = 192
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
        return (
            torch.tensor(r["state_seq"], dtype=torch.float32),
            torch.tensor(r["action"], dtype=torch.long),
        )


class XioiGRUOverfit(nn.Module):
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
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, ACTION_DIM * CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        z = out[:, -1, :]
        return self.head(z).view(-1, ACTION_DIM, CLASSES)


def load_rows() -> list[dict[str, Any]]:
    rows = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"Dataset vide: {DATASET_PATH}")
    return rows


def class_weights(rows: list[dict[str, Any]], device: torch.device) -> torch.Tensor:
    counts = torch.ones(CLASSES, dtype=torch.float32)
    for r in rows:
        for v in r["action"]:
            counts[int(v)] += 1
    inv = counts.sum() / counts
    weights = inv / inv.mean()
    # Overfit: on veut apprendre les activations non-zéro sans rendre 0 impossible.
    weights[0] *= 0.42
    weights = weights / weights.mean()
    return weights.to(device)


def evaluate(model: nn.Module, rows: list[dict[str, Any]], device: torch.device, criterion) -> dict[str, float]:
    loader = DataLoader(SeqDataset(rows), batch_size=BATCH_SIZE, shuffle=False)
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    exact = 0
    nonzero_total = 0
    nonzero_correct = 0
    pred_counter: Counter[int] = Counter()
    true_counter: Counter[int] = Counter()

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

            for v in pred.reshape(-1).detach().cpu().tolist():
                pred_counter[int(v)] += 1
            for v in y.reshape(-1).detach().cpu().tolist():
                true_counter[int(v)] += 1

    return {
        "loss": total_loss / max(1, len(rows)),
        "joint_acc": correct / max(1, total),
        "exact_acc": exact / max(1, len(rows)),
        "nonzero_acc": nonzero_correct / max(1, nonzero_total),
        "pred_counts": dict(pred_counter),
        "true_counts": dict(true_counter),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows()
    state_dim = len(rows[0]["state_seq"][0])
    seq_len = len(rows[0]["state_seq"])

    print("Device:", device)
    print("Dataset:", len(rows), "state_dim", state_dim, "seq_len", seq_len)
    print("Source:", rows[0].get("source"))

    loader = DataLoader(SeqDataset(rows), batch_size=BATCH_SIZE, shuffle=True)
    weights = class_weights(rows, device)
    print("Class weights:", [round(float(x), 4) for x in weights.detach().cpu()])

    model = XioiGRUOverfit(state_dim).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    best_metric = -1.0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total = 0
        correct = 0
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
            metric = metrics["exact_acc"] * 2.0 + metrics["nonzero_acc"] + metrics["joint_acc"]
            print(
                f"Epoch {epoch:03d} | train_loss={total_loss / max(1, len(rows)):.4f} "
                f"acc={correct / max(1, total):.4f} | "
                f"all_loss={metrics['loss']:.4f} joint={metrics['joint_acc']:.4f} "
                f"exact={metrics['exact_acc']:.4f} nonzero={metrics['nonzero_acc']:.4f}"
            )
            if metric > best_metric:
                best_metric = metric
                torch.save({
                    "version": 27,
                    "mode": "xioi_overfit",
                    "model_state": model.state_dict(),
                    "state_dim": state_dim,
                    "seq_len": seq_len,
                    "hidden": HIDDEN,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "action_dim": ACTION_DIM,
                    "classes": CLASSES,
                    "dataset": str(DATASET_PATH),
                    "metrics": metrics,
                }, MODEL_PATH)
                SUMMARY_PATH.write_text(json.dumps({
                    "version": 27,
                    "dataset_rows": len(rows),
                    "state_dim": state_dim,
                    "seq_len": seq_len,
                    "best_metric": best_metric,
                    "metrics": metrics,
                    "model": str(MODEL_PATH),
                }, indent=2), encoding="utf-8")
                print("  saved", MODEL_PATH)

    print("Done. Best metric:", best_metric)
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
