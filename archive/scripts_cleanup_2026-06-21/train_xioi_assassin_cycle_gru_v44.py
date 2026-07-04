#!/usr/bin/env python3
"""Train V44 GRU on explicit repeated cycle 70->295->70."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET = ROOT / "datasets" / "ml" / "xioi_assassin_cycle_v44_sequences.jsonl"
MODEL_PATH = ROOT / "models" / "xioi_assassin_cycle_gru_v44.pt"
SUMMARY_PATH = ROOT / "generated_replays" / "xioi_assassin_cycle_gru_v44_summary.json"

SEED = 44
BATCH_SIZE = 64
EPOCHS = 260
LR = 1e-3
HIDDEN = 192
LAYERS = 2
DROPOUT = 0.10

random.seed(SEED)
torch.manual_seed(SEED)


class ActionDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise RuntimeError(f"Empty dataset: {path}")
        self.state_dim = len(self.rows[0]["states"][0])
        self.seq_len = len(self.rows[0]["states"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        x = torch.tensor(r["states"], dtype=torch.float32)
        y = torch.tensor(r["action"], dtype=torch.long)
        return x, y


class GRUAction(nn.Module):
    def __init__(self, state_dim: int, hidden: int = HIDDEN, layers: int = LAYERS):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=DROPOUT if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 20 * 5),
        )

    def forward(self, x):
        y, _ = self.gru(x)
        z = y[:, -1, :]
        return self.head(z).view(-1, 20, 5)


def class_weights(ds: ActionDataset) -> torch.Tensor:
    c = Counter()
    for r in ds.rows:
        c.update(r["action"])
    total = sum(c.values())
    weights = []
    for i in range(5):
        # softened inverse frequency
        freq = c.get(i, 1) / max(1, total)
        weights.append((1.0 / max(freq, 1e-8)) ** 0.35)
    mean = sum(weights) / len(weights)
    weights = [w / mean for w in weights]
    print("Class counts:", dict(c))
    print("Class weights:", [round(w, 4) for w in weights])
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: nn.Module, ds: ActionDataset, device, weights) -> dict[str, float]:
    model.eval()
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    total_loss = 0.0
    total = 0
    correct = 0
    exact = 0
    nonzero_total = 0
    nonzero_correct = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, 5), y.reshape(-1), weight=weights.to(device))
        total_loss += float(loss.item()) * x.size(0)
        pred = logits.argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
        exact += int((pred == y).all(dim=1).sum().item())
        mask = y != 0
        nonzero_total += int(mask.sum().item())
        nonzero_correct += int(((pred == y) & mask).sum().item())
    return {
        "loss": total_loss / max(1, len(ds)),
        "joint": correct / max(1, total),
        "exact": exact / max(1, len(ds)),
        "nonzero": nonzero_correct / max(1, nonzero_total),
    }


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET}\nRun build_xioi_assassin_cycle_dataset_v44.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    ds = ActionDataset(DATASET)
    print("Dataset:", len(ds), "state_dim", ds.state_dim, "seq_len", ds.seq_len)
    weights = class_weights(ds).to(device)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    model = GRUAction(ds.state_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_metric = -1.0
    best_epoch = 0
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total = correct = exact = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 5), y.reshape(-1), weight=weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            exact += int((pred == y).all(dim=1).sum().item())

        ev = evaluate(model, ds, device, weights)
        metric = ev["joint"] + ev["exact"] + ev["nonzero"]
        should_print = epoch == 1 or epoch % 10 == 0 or epoch > EPOCHS - 5
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "state_dim": ds.state_dim,
                "seq_len": ds.seq_len,
                "hidden": HIDDEN,
                "layers": LAYERS,
                "dataset": str(DATASET),
                "epoch": epoch,
                "metric": metric,
                "version": 44,
            }, MODEL_PATH)
            if should_print:
                print("  saved", MODEL_PATH)
        if should_print:
            print(
                f"Epoch {epoch:03d} | loss={total_loss/max(1,len(ds)):.4f} "
                f"train_joint={correct/max(1,total):.4f} exact={exact/max(1,len(ds)):.4f} "
                f"all_joint={ev['joint']:.4f} nonzero={ev['nonzero']:.4f} all_exact={ev['exact']:.4f}"
            )
        if ev["joint"] >= 0.999 and ev["exact"] >= 0.99 and ev["nonzero"] >= 0.999:
            print("Early stop: near-perfect cycle imitation.")
            break

    summary = {
        "version": 44,
        "model": str(MODEL_PATH),
        "dataset": str(DATASET),
        "rows": len(ds),
        "best_epoch": best_epoch,
        "best_metric": best_metric,
        "state_dim": ds.state_dim,
        "seq_len": ds.seq_len,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
