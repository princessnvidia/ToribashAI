#!/usr/bin/env python3
import json
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

PROJECT = Path.home() / "Documents" / "ToribashAI"

DATASET_PATH = PROJECT / "datasets" / "ml" / "walk_fullbody_sequences_v1.jsonl"
MODEL_PATH = PROJECT / "models" / "walk_fullbody_gru_v1.pt"

SEQ_LEN = 8
NUM_JOINTS = 10
NUM_CLASSES = 5
INPUT_DIM = NUM_JOINTS

HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.20

BATCH_SIZE = 256
EPOCHS = 12
LR = 5e-4
VAL_RATIO = 0.10
SEED = 1234


class WalkDataset(Dataset):
    def __init__(self, path):
        self.rows = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        x = torch.tensor(row["input"], dtype=torch.float32)
        y = torch.tensor(row["target"], dtype=torch.long)

        return x, y


class WalkGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=INPUT_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(128, NUM_JOINTS * NUM_CLASSES),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits.view(-1, NUM_JOINTS, NUM_CLASSES)


def compute_class_weights(dataset):
    counts = Counter()

    for _, y in dataset:
        for v in y.tolist():
            counts[int(v)] += 1

    total = sum(counts.values())
    weights = torch.ones(NUM_CLASSES)

    for cls in range(NUM_CLASSES):
        c = counts.get(cls, 1)
        weights[cls] = (total / c) ** 0.25

    weights = weights / weights.mean()

    return weights, counts


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_items = 0
    correct = 0
    total = 0
    exact = 0
    exact_total = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(
                logits.reshape(-1, NUM_CLASSES),
                y.reshape(-1),
            )

            pred = logits.argmax(dim=-1)

            correct += (pred == y).sum().item()
            total += y.numel()

            exact += (pred == y).all(dim=1).sum().item()
            exact_total += y.shape[0]

            total_loss += loss.item() * x.shape[0]
            total_items += x.shape[0]

    return {
        "loss": total_loss / max(1, total_items),
        "joint_acc": correct / max(1, total),
        "exact_acc": exact / max(1, exact_total),
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = WalkDataset(DATASET_PATH)
    print("Dataset:", len(dataset))

    class_weights, counts = compute_class_weights(dataset)
    print("Class counts:", dict(sorted(counts.items())))
    print("Class weights:", class_weights.tolist())

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = WalkGRU().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_items = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(x)
            loss = criterion(
                logits.reshape(-1, NUM_CLASSES),
                y.reshape(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            pred = logits.argmax(dim=-1)

            correct += (pred == y).sum().item()
            total += y.numel()

            total_loss += loss.item() * x.shape[0]
            total_items += x.shape[0]

        train_loss = total_loss / max(1, total_items)
        train_acc = correct / max(1, total)

        val = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val['loss']:.4f} val_joint_acc={val['joint_acc']:.4f} "
            f"val_exact={val['exact_acc']:.4f}"
        )

        if val["loss"] < best_val:
            best_val = val["loss"]

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "seq_len": SEQ_LEN,
                    "num_joints": NUM_JOINTS,
                    "num_classes": NUM_CLASSES,
                    "control_joints": [
                        4, 5, 6, 7,
                        14, 15, 16,
                        17, 18, 19,
                    ],
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "class_weights": class_weights,
                    "class_counts": dict(counts),
                    "best_val_loss": best_val,
                },
                MODEL_PATH,
            )

            print("  saved:", MODEL_PATH)

    print("Done.")
    print("Best val loss:", best_val)
    print("Model:", MODEL_PATH)


if __name__ == "__main__":
    main()
