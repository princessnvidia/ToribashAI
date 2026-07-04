#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path.home() / "Documents/ToribashAI"
DATASET = ROOT / "datasets/ml/xioi_loop_len265_v52_sequences.jsonl"
MODEL = ROOT / "models/xioi_loop_len265_gru_v52.pt"
SUMMARY = ROOT / "generated_replays/xioi_loop_len265_gru_v52_summary.json"

SEQ_LEN = 8
STATE_DIM = 20
ACTION_DIM = 20
CLASSES = 5
HIDDEN = 192
LAYERS = 2
EPOCHS = 180
BATCH_SIZE = 64
LR = 1e-3


class SeqDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    def __len__(self):
        return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        x = torch.tensor(r["seq"], dtype=torch.float32) / 4.0
        y = torch.tensor(r["action"], dtype=torch.long)
        return x, y


class GRUAction(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(STATE_DIM, HIDDEN, num_layers=LAYERS, batch_first=True, dropout=0.10)
        self.head = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, ACTION_DIM * CLASSES))
    def forward(self, x):
        z, _ = self.gru(x)
        return self.head(z[:, -1]).view(-1, ACTION_DIM, CLASSES)


def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    exact = 0
    pred_counts = Counter()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).argmax(dim=-1)
            correct += (pred == y).sum().item()
            exact += (pred == y).all(dim=1).sum().item()
            total += y.numel()
            pred_counts.update(pred.cpu().flatten().tolist())
    n_rows = len(loader.dataset)
    return correct / max(1, total), exact / max(1, n_rows), pred_counts


def main():
    if not DATASET.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET}\nRun build_xioi_loop_len265_dataset_v52.py first.")
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SeqDataset(DATASET)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    counts = Counter()
    for _, y in ds:
        counts.update(y.tolist())
    total = sum(counts.values())
    weights = torch.ones(CLASSES)
    for c in range(CLASSES):
        weights[c] = (total / max(1, counts[c])) ** 0.25
    weights = weights / weights.mean()
    weights = weights.to(device)

    model = GRUAction().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    print("Device:", device)
    print("Dataset:", len(ds), "state_dim", STATE_DIM, "seq_len", SEQ_LEN)
    print("Class counts:", dict(counts))
    print("Class weights:", [round(float(x), 4) for x in weights.cpu()])

    best_metric = -1
    best_epoch = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, CLASSES), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item() * x.size(0)
        joint_acc, exact_acc, pred_counts = evaluate(model, loader, device)
        metric = joint_acc + exact_acc
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
            }, MODEL)
            print("  saved", MODEL)
        if epoch == 1 or epoch % 10 == 0 or epoch > EPOCHS - 5:
            print(f"Epoch {epoch:03d} | loss={loss_sum/len(ds):.4f} joint={joint_acc:.4f} exact={exact_acc:.4f} pred={pred_counts.most_common()}")

    summary = {"version": 52, "model": str(MODEL), "dataset": str(DATASET), "rows": len(ds), "best_epoch": best_epoch, "best_metric": best_metric}
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
