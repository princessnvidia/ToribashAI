#!/usr/bin/env python3
from pathlib import Path
import json
import random
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

BASE = Path.home() / "Documents/ToribashAI"

DATASET = BASE / "datasets" / "ml" / "parkour_sequences_len8.jsonl"
MODELS = BASE / "models"
MODEL_PATH = MODELS / "parkour_gru_v1.pt"
SUMMARY_PATH = MODELS / "parkour_gru_v1_summary.json"

SEQ_LEN = 8
STATE_DIM = 273
JOINTS = 20
CLASSES = 5

BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3

HIDDEN_SIZE = 256
NUM_LAYERS = 2
DROPOUT = 0.15

WEIGHT_POWER = 0.25

torch.manual_seed(42)
random.seed(42)


class ParkourSequenceDataset(Dataset):
    def __init__(self, path):
        self.states = []
        self.actions = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                states = row["states"]
                action = row["action"]

                if len(states) != SEQ_LEN:
                    continue

                if any(len(s) != STATE_DIM for s in states):
                    continue

                if len(action) != JOINTS:
                    continue

                self.states.append(states)
                self.actions.append(action)

        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.actions = torch.tensor(self.actions, dtype=torch.long)

        mean = self.states.reshape(-1, STATE_DIM).mean(dim=0, keepdim=True)
        std = self.states.reshape(-1, STATE_DIM).std(dim=0, keepdim=True)
        std[std < 1e-6] = 1.0

        self.mean = mean
        self.std = std
        self.states = (self.states - mean.view(1, 1, -1)) / std.view(1, 1, -1)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

    def class_counts(self):
        flat = self.actions.reshape(-1).tolist()
        counts = Counter(flat)
        return [counts.get(i, 0) for i in range(CLASSES)]


class GRUPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=STATE_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Linear(HIDDEN_SIZE, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, JOINTS * CLASSES),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits.view(-1, JOINTS, CLASSES)


def make_class_weights(class_counts, device):
    counts = torch.tensor(class_counts, dtype=torch.float32, device=device)
    counts = torch.clamp(counts, min=1.0)

    inverse = counts.sum() / (CLASSES * counts)
    weights = torch.pow(inverse, WEIGHT_POWER)
    weights = weights / weights.mean()

    return weights


def joint_accuracy(logits, y):
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def exact_action_accuracy(logits, y):
    pred = logits.argmax(dim=-1)
    return (pred == y).all(dim=1).float().mean().item()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_joint_acc = 0.0
    total_exact_acc = 0.0
    batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(
            logits.reshape(-1, CLASSES),
            y.reshape(-1)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_joint_acc += joint_accuracy(logits, y)
        total_exact_acc += exact_action_accuracy(logits, y)
        batches += 1

    return {
        "loss": total_loss / batches,
        "joint_acc": total_joint_acc / batches,
        "exact_acc": total_exact_acc / batches,
    }


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_joint_acc = 0.0
    total_exact_acc = 0.0
    batches = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(
            logits.reshape(-1, CLASSES),
            y.reshape(-1)
        )

        total_loss += loss.item()
        total_joint_acc += joint_accuracy(logits, y)
        total_exact_acc += exact_action_accuracy(logits, y)
        batches += 1

    return {
        "loss": total_loss / batches,
        "joint_acc": total_joint_acc / batches,
        "exact_acc": total_exact_acc / batches,
    }


def main():
    MODELS.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Dataset:", DATASET)

    dataset = ParkourSequenceDataset(DATASET)

    print("Sequences:", len(dataset))

    class_counts = dataset.class_counts()
    class_weights = make_class_weights(class_counts, device)

    print("Class counts:", class_counts)
    print("Class weights:", class_weights.detach().cpu().tolist())

    train_size = int(len(dataset) * 0.9)
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = GRUPolicy().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        val_metrics = eval_epoch(
            model,
            val_loader,
            criterion,
            device
        )

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train loss {train_metrics['loss']:.4f} | "
            f"train joint acc {train_metrics['joint_acc']:.4f} | "
            f"train exact {train_metrics['exact_acc']:.4f} | "
            f"val loss {val_metrics['loss']:.4f} | "
            f"val joint acc {val_metrics['joint_acc']:.4f} | "
            f"val exact {val_metrics['exact_acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "mean": dataset.mean,
                    "std": dataset.std,
                    "seq_len": SEQ_LEN,
                    "state_dim": STATE_DIM,
                    "joints": JOINTS,
                    "classes": CLASSES,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "class_counts": class_counts,
                    "class_weights": class_weights.detach().cpu(),
                    "weight_power": WEIGHT_POWER,
                    "history": history,
                },
                MODEL_PATH
            )

    SUMMARY_PATH.write_text(
        json.dumps(
            {
                "dataset": str(DATASET),
                "model": str(MODEL_PATH),
                "device": device,
                "sequences": len(dataset),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "seq_len": SEQ_LEN,
                "state_dim": STATE_DIM,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
                "class_counts": class_counts,
                "class_weights": class_weights.detach().cpu().tolist(),
                "weight_power": WEIGHT_POWER,
                "history": history,
                "best_val_loss": best_val_loss,
            },
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\nTerminé.")
    print("Modèle:", MODEL_PATH)
    print("Résumé:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
