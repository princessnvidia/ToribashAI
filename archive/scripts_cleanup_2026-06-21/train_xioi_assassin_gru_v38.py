#!/usr/bin/env python3
"""Train a small overfit GRU on the V38 Xioi assassin walk dataset."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET = ROOT / "datasets/ml/xioi_assassin_walk_v38_sequences.jsonl"
MODEL_OUT = ROOT / "models/xioi_assassin_gru_v38.pt"
SUMMARY_OUT = ROOT / "generated_replays/xioi_assassin_gru_v38_train_summary.json"

HIDDEN = 160
LAYERS = 2
EPOCHS = 350
BATCH = 32
LR = 8e-4


class WalkDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not self.rows:
            raise RuntimeError(f"Empty dataset: {path}")
        self.state_dim = len(self.rows[0]["state_seq"][0])
        self.seq_len = len(self.rows[0]["state_seq"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return (
            torch.tensor(r["state_seq"], dtype=torch.float32),
            torch.tensor(r["action"], dtype=torch.long),
        )


class GRUAction(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.gru = nn.GRU(state_dim, HIDDEN, num_layers=LAYERS, batch_first=True, dropout=0.10)
        self.head = nn.Sequential(nn.LayerNorm(HIDDEN), nn.Linear(HIDDEN, 20 * 5))

    def forward(self, x):
        y, _ = self.gru(x)
        z = y[:, -1, :]
        return self.head(z).view(-1, 20, 5)


def main() -> None:
    if not DATASET.exists():
        raise FileNotFoundError(f"Missing {DATASET}. Run build_xioi_assassin_dataset_v38.py first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = WalkDataset(DATASET)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)

    counts = Counter()
    for _, y in ds:
        counts.update(int(v) for v in y.tolist())
    total = sum(counts.values())
    weights = []
    for i in range(5):
        c = max(1, counts.get(i, 0))
        weights.append((total / c) ** 0.35)
    weights_t = torch.tensor(weights, dtype=torch.float32, device=device)
    weights_t = weights_t / weights_t.mean()

    model = GRUAction(ds.state_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights_t)

    best_metric = -1.0
    best_epoch = 0
    print("Device:", device)
    print("Dataset:", len(ds), "state_dim", ds.state_dim, "seq_len", ds.seq_len)
    print("Class counts:", dict(counts))
    print("Class weights:", [round(float(x), 4) for x in weights_t.cpu()])

    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        correct = totalj = exact = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, 5), y.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
            pred = logits.argmax(dim=-1)
            correct += int((pred == y).sum().item())
            totalj += y.numel()
            exact += int((pred == y).all(dim=1).sum().item())

        model.eval()
        with torch.no_grad():
            all_x = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
            all_y = torch.stack([ds[i][1] for i in range(len(ds))]).to(device)
            logits = model(all_x)
            pred = logits.argmax(dim=-1)
            joint_acc = float((pred == all_y).float().mean().item())
            exact_acc = float((pred == all_y).all(dim=1).float().mean().item())
            nonzero_mask = all_y != 0
            nonzero_acc = float((pred[nonzero_mask] == all_y[nonzero_mask]).float().mean().item()) if nonzero_mask.any() else 0.0
            metric = joint_acc + nonzero_acc + exact_acc * 2.0

        if epoch % 10 == 0 or epoch == 1 or epoch > EPOCHS - 5:
            print(
                f"Epoch {epoch:03d} | loss={sum(losses)/len(losses):.4f} "
                f"train_joint={correct/max(1,totalj):.4f} exact={exact/max(1,len(ds)):.4f} "
                f"all_joint={joint_acc:.4f} nonzero={nonzero_acc:.4f} all_exact={exact_acc:.4f}"
            )

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
            if epoch % 10 == 0 or epoch == 1:
                print("  saved", MODEL_OUT)

    summary = {"model": str(MODEL_OUT), "best_epoch": best_epoch, "best_metric": best_metric, "rows": len(ds)}
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Done:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
