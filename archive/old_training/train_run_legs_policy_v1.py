#!/usr/bin/env python3
from pathlib import Path
import json
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT = Path.home() / "Documents" / "ToribashAI"

DATA_PATH = PROJECT / "datasets" / "locomotion" / "run_legs_dataset_v1.jsonl"
OUT_MODEL = PROJECT / "models" / "run_legs_policy_v1.pt"
OUT_INFO = PROJECT / "models" / "run_legs_policy_v1_info.json"

SEED = 42
BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-3

NUM_LEG_JOINTS = 6
NUM_CLASSES = 5


class RunLegsDataset(Dataset):
    def __init__(self, rows, mean=None, std=None):
        self.x = []
        self.y = []

        for row in rows:
            f = row["features"]

            features = [
                float(f["forward_speed"]),
                float(f["leg_activity"]),
                float(f["support_change_rate"]),
                float(f["forward_lean"]),
                float(f["z_min"]),
                float(f["z_range"]),
            ]

            # état jambes actuel + features globales
            self.x.append([float(v) for v in row["leg_now"]] + features)
            self.y.append([int(v) for v in row["leg_next"]])

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

        if mean is None:
            self.mean = self.x.mean(dim=0)
            self.std = self.x.std(dim=0).clamp_min(1e-6)
        else:
            self.mean = mean
            self.std = std

        self.x = (self.x - self.mean) / self.std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class RunLegsPolicy(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(96, NUM_LEG_JOINTS * NUM_CLASSES),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits.view(-1, NUM_LEG_JOINTS, NUM_CLASSES)


def load_rows():
    rows = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_rows(rows):
    random.seed(SEED)
    rows = list(rows)
    random.shuffle(rows)
    n = len(rows)
    return rows[: int(n * 0.8)], rows[int(n * 0.8):]


def evaluate(model, loader, device):
    model.eval()

    total = 0
    correct = 0
    exact = 0

    pred_counts = Counter()
    true_counts = Counter()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = logits.argmax(dim=-1)

            correct += (pred == y).sum().item()
            total += y.numel()
            exact += (pred == y).all(dim=1).sum().item()

            for v in pred.cpu().reshape(-1).tolist():
                pred_counts[str(int(v))] += 1
            for v in y.cpu().reshape(-1).tolist():
                true_counts[str(int(v))] += 1

    return {
        "joint_acc": correct / max(1, total),
        "exact_acc": exact / max(1, len(loader.dataset)),
        "pred_counts": dict(pred_counts),
        "true_counts": dict(true_counts),
    }


def main():
    rows = load_rows()
    if not rows:
        raise RuntimeError(f"Dataset vide: {DATA_PATH}")

    train_rows, val_rows = split_rows(rows)

    train_ds = RunLegsDataset(train_rows)
    val_ds = RunLegsDataset(val_rows, mean=train_ds.mean, std=train_ds.std)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Rows:", len(rows))
    print("Train:", len(train_ds))
    print("Val:", len(val_ds))
    print("Device:", device)

    global_counts = Counter()
    for row in train_rows:
        for v in row["leg_next"]:
            global_counts[str(int(v))] += 1

    weights = torch.tensor(
        [1.0 / max(1, global_counts[str(i)]) for i in range(NUM_CLASSES)],
        dtype=torch.float32,
        device=device,
    )
    weights = weights / weights.mean()

    print("Class counts:", dict(global_counts))
    print("Class weights:", [round(float(w), 4) for w in weights.cpu().tolist()])

    model = RunLegsPolicy(input_dim=train_ds.x.shape[1]).to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_score = -1.0
    best_eval = None

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)

            loss = 0.0
            for j in range(NUM_LEG_JOINTS):
                loss = loss + criterion(logits[:, j, :], y[:, j])

            loss = loss / NUM_LEG_JOINTS
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            batches += 1

        val_eval = evaluate(model, val_loader, device)

        score = val_eval["joint_acc"] + val_eval["exact_acc"]

        print(
            f"Epoch {epoch:02d} | "
            f"loss={total_loss / max(1, batches):.4f} | "
            f"val_joint_acc={val_eval['joint_acc']:.4f} | "
            f"val_exact_acc={val_eval['exact_acc']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_eval = val_eval

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "mean": train_ds.mean,
                    "std": train_ds.std,
                    "input_dim": train_ds.x.shape[1],
                    "num_leg_joints": NUM_LEG_JOINTS,
                    "num_classes": NUM_CLASSES,
                    "leg_joints": [14, 15, 16, 17, 18, 19],
                    "best_eval": best_eval,
                },
                OUT_MODEL,
            )

            OUT_INFO.write_text(
                json.dumps(
                    {
                        "dataset": str(DATA_PATH),
                        "rows": len(rows),
                        "train": len(train_ds),
                        "val": len(val_ds),
                        "class_counts": dict(global_counts),
                        "class_weights": [float(w) for w in weights.cpu().tolist()],
                        "best_eval": best_eval,
                        "model": str(OUT_MODEL),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    print()
    print("Terminé.")
    print(f"Model: {OUT_MODEL}")
    print(f"Info: {OUT_INFO}")
    print("Best eval:", best_eval)


if __name__ == "__main__":
    main()
