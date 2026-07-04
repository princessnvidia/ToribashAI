#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET = ROOT / "datasets" / "ml" / "xioi_assassin_loop_v42_sequences.jsonl"
MODEL_OUT = ROOT / "models" / "xioi_assassin_loop_gru_v42.pt"
SUMMARY = ROOT / "generated_replays" / "xioi_assassin_loop_gru_v42_summary.json"

HIDDEN = 192
LAYERS = 2
BATCH = 32
EPOCHS = 320
LR = 2e-3


class SeqDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise RuntimeError(f"Empty dataset: {path}")
        self.state_dim = len(self.rows[0]["sequence"][0])
        self.seq_len = len(self.rows[0]["sequence"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        x = torch.tensor(r["sequence"], dtype=torch.float32)
        y = torch.tensor(r["action"], dtype=torch.long)
        return x, y


class GRUAction(nn.Module):
    def __init__(self, state_dim: int, hidden: int = HIDDEN, layers: int = LAYERS):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden, layers, batch_first=True, dropout=0.10 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 20 * 5))

    def forward(self, x):
        z, _ = self.gru(x)
        z = z[:, -1]
        return self.head(z).view(-1, 20, 5)


@torch.no_grad()
def eval_all(model, ds, device):
    loader = DataLoader(ds, batch_size=128, shuffle=False)
    total = correct = exact = nonzero_correct = nonzero_total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
        exact += (pred == y).all(dim=1).sum().item()
        nz = y != 0
        nonzero_correct += ((pred == y) & nz).sum().item()
        nonzero_total += nz.sum().item()
    return {
        "joint": correct / max(1, total),
        "exact": exact / max(1, len(ds)),
        "nonzero": nonzero_correct / max(1, nonzero_total),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = SeqDataset(DATASET)
    print("Device:", device)
    print("Dataset:", len(ds), "state_dim", ds.state_dim, "seq_len", ds.seq_len)

    counts = Counter()
    for r in ds.rows:
        counts.update(int(v) for v in r["action"])
    print("Class counts:", dict(counts))
    raw = torch.tensor([1.0 / max(1, counts.get(i, 0)) for i in range(5)], dtype=torch.float32)
    weights = raw / raw.mean()
    weights = torch.sqrt(weights).to(device)
    print("Class weights:", [round(float(x), 4) for x in weights.cpu()])

    model = GRUAction(ds.state_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)

    best_metric = -1.0
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total = correct = exact = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, 5), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.numel()
            exact += (pred == y).all(dim=1).sum().item()
        ev = eval_all(model, ds, device)
        metric = ev["joint"] + ev["exact"] + ev["nonzero"]
        if metric > best_metric:
            best_metric = metric
            best_epoch = epoch
            MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "state_dim": ds.state_dim,
                "seq_len": ds.seq_len,
                "hidden": HIDDEN,
                "layers": LAYERS,
                "dataset": str(DATASET),
                "epoch": epoch,
                "metric": metric,
            }, MODEL_OUT)
            saved = " saved"
        else:
            saved = ""
        if epoch <= 3 or epoch % 10 == 0 or epoch > EPOCHS - 5 or saved:
            print(
                f"Epoch {epoch:03d} | loss={total_loss/len(ds):.4f} "
                f"train_joint={correct/max(1,total):.4f} exact={exact/max(1,len(ds)):.4f} "
                f"all_joint={ev['joint']:.4f} nonzero={ev['nonzero']:.4f} all_exact={ev['exact']:.4f}" + saved
            )
    summary = {"model": str(MODEL_OUT), "best_epoch": best_epoch, "best_metric": best_metric, "rows": len(ds)}
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
