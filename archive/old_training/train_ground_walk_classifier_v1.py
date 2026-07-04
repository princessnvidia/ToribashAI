#!/usr/bin/env python3
from pathlib import Path
import json
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT = Path.home() / "Documents" / "ToribashAI"

DATA_PATH = PROJECT / "datasets" / "locomotion" / "ground_walk_dataset_v1.jsonl"
OUT_MODEL = PROJECT / "models" / "ground_walk_classifier_v1.pt"
OUT_INFO = PROJECT / "models" / "ground_walk_classifier_v1_info.json"

SEED = 42
BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-3

FEATURE_KEYS = [
    "forward_speed",
    "leg_activity",
    "core_activity",
    "support_change_rate",
    "forward_lean",
    "z_min",
    "z_range",
]


class GroundWalkDataset(Dataset):
    def __init__(self, rows, label_to_id, mean=None, std=None):
        self.x = []
        self.y = []

        for row in rows:
            f = row["features"]

            self.x.append([
                float(row["forward_speed"]),
                float(f["leg_activity"]),
                float(f["core_activity"]),
                float(f["support_change_rate"]),
                float(f["forward_lean"]),
                float(f["z_min"]),
                float(f["z_range"]),
            ])

            self.y.append(label_to_id[row["label"]])

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


def evaluate(model, loader, device, id_to_label):
    model.eval()

    correct = 0
    total = 0

    confusion = {
        label: Counter()
        for label in id_to_label.values()
    }

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.numel()

            for true_id, pred_id in zip(y.cpu().tolist(), pred.cpu().tolist()):
                true_label = id_to_label[true_id]
                pred_label = id_to_label[pred_id]
                confusion[true_label][pred_label] += 1

    return correct / max(1, total), confusion


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset introuvable: {DATA_PATH}")

    rows = load_rows()

    if not rows:
        raise RuntimeError("Dataset vide.")

    label_counts = Counter(row["label"] for row in rows)
    labels = sorted(label_counts.keys())

    label_to_id = {label: i for i, label in enumerate(labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}

    print("Rows:", len(rows))
    print("Labels:", label_counts)
    print("Label map:", label_to_id)

    train_rows, val_rows = split_rows(rows)

    train_ds = GroundWalkDataset(train_rows, label_to_id)
    val_ds = GroundWalkDataset(
        val_rows,
        label_to_id,
        mean=train_ds.mean,
        std=train_ds.std,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = MLP(len(FEATURE_KEYS), len(labels)).to(device)

    train_counts = Counter(row["label"] for row in train_rows)

    weights = torch.tensor(
        [
            1.0 / max(1, train_counts[id_to_label[i]])
            for i in range(len(labels))
        ],
        dtype=torch.float32,
        device=device,
    )

    weights = weights / weights.mean()

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_acc = 0.0
    best_confusion = None

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
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1

        train_acc, _ = evaluate(model, train_loader, device, id_to_label)
        val_acc, confusion = evaluate(model, val_loader, device, id_to_label)

        print(
            f"Epoch {epoch:02d} | "
            f"loss={total_loss / max(1, batches):.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_confusion = confusion

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "feature_keys": FEATURE_KEYS,
                    "label_to_id": label_to_id,
                    "id_to_label": id_to_label,
                    "mean": train_ds.mean,
                    "std": train_ds.std,
                    "input_dim": len(FEATURE_KEYS),
                    "output_dim": len(labels),
                    "best_val_acc": best_val_acc,
                },
                OUT_MODEL,
            )

            OUT_INFO.write_text(
                json.dumps(
                    {
                        "dataset": str(DATA_PATH),
                        "rows": len(rows),
                        "label_counts": dict(label_counts),
                        "feature_keys": FEATURE_KEYS,
                        "label_to_id": label_to_id,
                        "id_to_label": id_to_label,
                        "best_val_acc": best_val_acc,
                        "best_confusion": {
                            k: dict(v)
                            for k, v in best_confusion.items()
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    print()
    print("Terminé.")
    print(f"Best val acc: {best_val_acc:.4f}")
    print(f"Model: {OUT_MODEL}")
    print(f"Info: {OUT_INFO}")

    if best_confusion is not None:
        print()
        print("Best confusion:")
        for true_label, preds in best_confusion.items():
            print(true_label, dict(preds))


if __name__ == "__main__":
    main()
