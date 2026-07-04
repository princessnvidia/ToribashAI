#!/usr/bin/env python3
from pathlib import Path
import json
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT = Path.home() / "Documents" / "ToribashAI"

DATA_PATH = PROJECT / "datasets" / "motion_patterns" / "motion_patterns_v1.jsonl"
OUT_MODEL = PROJECT / "models" / "motion_pattern_classifier_v1.pt"
OUT_LABELS = PROJECT / "models" / "motion_pattern_classifier_v1_labels.json"

BATCH_SIZE = 128
EPOCHS = 25
LR = 1e-3
SEED = 42

FEATURE_KEYS = [
    "delta_x",
    "delta_y",
    "delta_z",
    "displacement_xy",
    "speed_xy",
    "z_min",
    "z_max",
    "z_mean",
    "z_range",
    "activity",
    "leg_activity",
    "arm_activity",
    "core_activity",
    "action_change_rate",
]


class MotionDataset(Dataset):
    def __init__(self, rows, label_to_id):
        self.x = []
        self.y = []

        for row in rows:
            f = row["features"]
            self.x.append([float(f[k]) for k in FEATURE_KEYS])
            self.y.append(label_to_id[row["rough_label"]])

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

        self.mean = self.x.mean(dim=0)
        self.std = self.x.std(dim=0).clamp_min(1e-6)
        self.x = (self.x - self.mean) / self.std

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


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
    train_n = int(n * 0.8)

    return rows[:train_n], rows[train_n:]


def accuracy(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

    return correct / max(1, total)


def main():
    rows = load_rows()
    if not rows:
        raise RuntimeError(f"Aucune donnée dans {DATA_PATH}")

    labels = sorted(set(row["rough_label"] for row in rows))
    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    print("Rows:", len(rows))
    print("Labels:", Counter(row["rough_label"] for row in rows))
    print("Label map:", label_to_id)

    train_rows, val_rows = split_rows(rows)

    train_ds = MotionDataset(train_rows, label_to_id)
    val_ds = MotionDataset(val_rows, label_to_id)

    # important : même normalisation train sur val
    val_ds.mean = train_ds.mean
    val_ds.std = train_ds.std
    val_ds.x = (val_ds.x * val_ds.std + val_ds.mean - train_ds.mean) / train_ds.std

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = MLP(len(FEATURE_KEYS), len(labels)).to(device)

    counts = Counter(row["rough_label"] for row in train_rows)
    weights = torch.tensor(
        [1.0 / max(1, counts[id_to_label[i]]) for i in range(len(labels))],
        dtype=torch.float32,
        device=device,
    )
    weights = weights / weights.mean()

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        train_acc = accuracy(model, train_loader, device)
        val_acc = accuracy(model, val_loader, device)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={total_loss / max(1, batches):.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc > best_val:
            best_val = val_acc
            OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                "model_state": model.state_dict(),
                "feature_keys": FEATURE_KEYS,
                "label_to_id": label_to_id,
                "id_to_label": id_to_label,
                "mean": train_ds.mean,
                "std": train_ds.std,
                "input_dim": len(FEATURE_KEYS),
                "output_dim": len(labels),
                "best_val_acc": best_val,
            }, OUT_MODEL)

            OUT_LABELS.write_text(
                json.dumps({
                    "feature_keys": FEATURE_KEYS,
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "best_val_acc": best_val,
                }, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    print()
    print("Terminé.")
    print(f"Best val acc: {best_val:.4f}")
    print(f"Model: {OUT_MODEL}")
    print(f"Labels: {OUT_LABELS}")


if __name__ == "__main__":
    main()
