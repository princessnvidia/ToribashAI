#!/usr/bin/env python3
from pathlib import Path
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

BASE = Path.home() / "Documents/ToribashAI"

DATASET = BASE / "datasets" / "ml" / "parkour_transitions_clean.jsonl"
MODELS = BASE / "models"
MODEL_PATH = MODELS / "parkour_mlp_v1.pt"
SUMMARY_PATH = MODELS / "parkour_mlp_v1_summary.json"

STATE_DIM = 273
JOINTS = 20
CLASSES = 5

BATCH_SIZE = 256
EPOCHS = 20
LR = 1e-3

torch.manual_seed(42)
random.seed(42)


class ParkourDataset(Dataset):
    def __init__(self, path):
        self.states = []
        self.actions = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)

                state = row["state"]
                action = row["action"]

                if len(state) != STATE_DIM:
                    continue

                if len(action) != JOINTS:
                    continue

                self.states.append(state)
                self.actions.append(action)

        self.states = torch.tensor(self.states, dtype=torch.float32)
        self.actions = torch.tensor(self.actions, dtype=torch.long)

        mean = self.states.mean(dim=0, keepdim=True)
        std = self.states.std(dim=0, keepdim=True)
        std[std < 1e-6] = 1.0

        self.mean = mean
        self.std = std
        self.states = (self.states - mean) / std

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]


class MLPPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, JOINTS * CLASSES),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits.view(-1, JOINTS, CLASSES)


def joint_accuracy(logits, y):
    pred = logits.argmax(dim=-1)
    correct = (pred == y).float()
    return correct.mean().item()


def exact_action_accuracy(logits, y):
    pred = logits.argmax(dim=-1)
    exact = (pred == y).all(dim=1).float()
    return exact.mean().item()


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

    dataset = ParkourDataset(DATASET)

    print("Transitions:", len(dataset))

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

    model = MLPPolicy().to(device)

    criterion = nn.CrossEntropyLoss()
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
                    "state_dim": STATE_DIM,
                    "joints": JOINTS,
                    "classes": CLASSES,
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
                "transitions": len(dataset),
                "epochs": EPOCHS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
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
