#!/usr/bin/env python3
"""
train_xioi_assassin_cycle_gru_v45.py

Train a GRU on the real action cycle dataset created by V45.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET = ROOT / "datasets" / "ml" / "xioi_assassin_cycle_v45_sequences.jsonl"
MODEL_OUT = ROOT / "models" / "xioi_assassin_cycle_gru_v45.pt"
SUMMARY_OUT = ROOT / "generated_replays" / "xioi_assassin_cycle_gru_v45_summary.json"

SEQ_LEN = 8
ACTION_DIM = 20
STATE_DIM = 20
HIDDEN = 192
LAYERS = 2
EPOCHS = 260
BATCH_SIZE = 64
LR = 3e-4


class ActionDataset(Dataset):
    def __init__(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset: {path}\nRun build_xioi_assassin_cycle_dataset_v45.py first.")
        self.rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        x = torch.tensor(r["seq"], dtype=torch.float32) / 4.0
        y = torch.tensor(r["target"], dtype=torch.long)
        return x, y


class GRUAction(nn.Module):
    def __init__(self, state_dim=STATE_DIM, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden, num_layers=layers, batch_first=True, dropout=0.05 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, ACTION_DIM * 5))

    def forward(self, x):
        out, _ = self.gru(x)
        z = out[:, -1]
        return self.head(z).view(-1, ACTION_DIM, 5)


def evaluate(model, loader, device):
    model.eval()
    total = 0
    joint_ok = 0
    exact_ok = 0
    nonzero_total = 0
    nonzero_ok = 0
    pred_counts = Counter()
    true_counts = Counter()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            joint_ok += (pred == y).sum().item()
            total += y.numel()
            exact_ok += (pred == y).all(dim=1).sum().item()
            mask = y != 0
            nonzero_total += mask.sum().item()
            nonzero_ok += ((pred == y) & mask).sum().item()
            for v in pred.cpu().reshape(-1).tolist():
                pred_counts[int(v)] += 1
            for v in y.cpu().reshape(-1).tolist():
                true_counts[int(v)] += 1
    n = len(loader.dataset)
    return {
        "joint": joint_ok / max(1, total),
        "exact": exact_ok / max(1, n),
        "nonzero": nonzero_ok / max(1, nonzero_total),
        "pred_counts": pred_counts.most_common(),
        "true_counts": true_counts.most_common(),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = ActionDataset(DATASET)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    eval_loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    counts = Counter()
    for r in ds.rows:
        counts.update(r["target"])
    print("Device:", device)
    print("Dataset:", len(ds), "state_dim", STATE_DIM, "seq_len", SEQ_LEN)
    print("Class counts:", dict(counts))

    weights = torch.ones(5, dtype=torch.float32)
    total = sum(counts.values())
    for k in range(5):
        if counts[k] > 0:
            weights[k] = (total / (5 * counts[k])) ** 0.35
        else:
            weights[k] = 1.5
    print("Class weights:", [round(float(x), 4) for x in weights])

    model = GRUAction().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device))

    best_metric = -1.0
    best_epoch = 0
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, 5), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        metrics = evaluate(model, eval_loader, device)
        metric = metrics["joint"] + metrics["exact"] + metrics["nonzero"]
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            torch.save({
                "model_state": model.state_dict(),
                "state_dim": STATE_DIM,
                "seq_len": SEQ_LEN,
                "hidden": HIDDEN,
                "layers": LAYERS,
                "dataset": str(DATASET),
                "epoch": epoch,
                "metric": metric,
            }, MODEL_OUT)
            if epoch == 1 or epoch % 10 == 0:
                print("  saved", MODEL_OUT)

        if epoch == 1 or epoch % 10 == 0 or epoch > EPOCHS - 5:
            print(
                f"Epoch {epoch:03d} | loss={total_loss/max(1,len(loader)):.4f} "
                f"joint={metrics['joint']:.4f} exact={metrics['exact']:.4f} nonzero={metrics['nonzero']:.4f} "
                f"pred={metrics['pred_counts'][:5]}"
            )

    summary = {"version": 45, "model": str(MODEL_OUT), "dataset": str(DATASET), "rows": len(ds), "best_epoch": best_epoch, "best_metric": best_metric}
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
