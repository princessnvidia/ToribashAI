#!/usr/bin/env python3
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

ROOT = Path.home() / "Documents" / "ToribashAI"

DATASET_PATH = ROOT / "datasets/ml/walk_motion_v2.jsonl"
MODEL_PATH = ROOT / "models/walk_fullbody_gru_v3_motion.pt"
SUMMARY_PATH = ROOT / "models/walk_fullbody_gru_v3_motion_summary.json"
EPOCHS = 24
BATCH_SIZE = 64

SEED = 42
LR = 5e-4
VAL_RATIO = 0.12

HIDDEN_SIZE = 160
NUM_LAYERS = 1
DROPOUT = 0.20

NUM_CLASSES = 5
ACTION_DIM = 20


random.seed(SEED)
torch.manual_seed(SEED)


def extract_state_sequence(ex):
    for key in ["states", "state_seq", "sequence", "x"]:
        if key in ex and isinstance(ex[key], list):
            return ex[key]
    raise KeyError("Aucune séquence d'état trouvée.")


def extract_action(ex):
    for key in ["target_action", "action", "y"]:
        if key in ex and isinstance(ex[key], list):
            return ex[key]

    if "actions" in ex and isinstance(ex["actions"], list) and ex["actions"]:
        return ex["actions"][-1]

    raise KeyError("Aucune action trouvée.")


class WalkQualityDataset(Dataset):
    def __init__(self, path):
        self.items = []
        self.state_dim = None

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                ex = json.loads(line)
                states = extract_state_sequence(ex)
                action = extract_action(ex)

                if len(action) != ACTION_DIM:
                    continue

                if self.state_dim is None:
                    self.state_dim = len(states[0])

                self.items.append((states, action, ex.get("walk_motion_score", 1.0)))

        if not self.items:
            raise RuntimeError("Dataset vide.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        states, action, quality = self.items[idx]

        x = torch.tensor(states, dtype=torch.float32)
        y = torch.tensor(action, dtype=torch.long)
        q = torch.tensor(float(quality), dtype=torch.float32)

        return x, y, q


class WalkGRU(nn.Module):
    def __init__(self, state_dim):
        super().__init__()

        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, 256),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, ACTION_DIM * NUM_CLASSES),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits.view(-1, ACTION_DIM, NUM_CLASSES)


def compute_class_weights(dataset):
    counts = torch.ones(NUM_CLASSES)

    for _, action, _ in dataset:
        for v in action.tolist():
            if 0 <= v < NUM_CLASSES:
                counts[v] += 1

    inv = counts.sum() / counts
    weights = inv ** 0.35
    weights = weights / weights.mean()

    return weights


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_joints = 0
    correct_joints = 0
    exact_actions = 0
    total_actions = 0

    pred_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)
    true_counts = torch.zeros(NUM_CLASSES, dtype=torch.long)

    with torch.no_grad():
        for x, y, q in loader:
            x = x.to(device)
            y = y.to(device)
            q = q.to(device)

            logits = model(x)

            raw_loss = criterion(
                logits.reshape(-1, NUM_CLASSES),
                y.reshape(-1),
            ).view(y.shape)

            quality_weight = 0.75 + q.view(-1, 1) * 1.5
            loss = (raw_loss * quality_weight).mean()

            pred = logits.argmax(dim=-1)

            correct_joints += (pred == y).sum().item()
            total_joints += y.numel()

            exact_actions += (pred == y).all(dim=1).sum().item()
            total_actions += y.shape[0]

            total_loss += loss.item() * y.shape[0]

            for c in range(NUM_CLASSES):
                pred_counts[c] += (pred == c).sum().item()
                true_counts[c] += (y == c).sum().item()

    return {
        "loss": total_loss / max(total_actions, 1),
        "joint_acc": correct_joints / max(total_joints, 1),
        "exact_acc": exact_actions / max(total_actions, 1),
        "pred_counts": pred_counts.tolist(),
        "true_counts": true_counts.tolist(),
    }


def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    dataset = WalkQualityDataset(DATASET_PATH)
    print("Dataset:", len(dataset))
    print("State dim:", dataset.state_dim)

    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(dataset).to(device)
    print("Class weights:", class_weights.cpu().tolist())

    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    model = WalkGRU(dataset.state_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_loss = float("inf")
    best_epoch = -1
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        total_loss = 0.0
        total_items = 0

        for x, y, q in train_loader:
            x = x.to(device)
            y = y.to(device)
            q = q.to(device)

            optimizer.zero_grad()

            logits = model(x)

            raw_loss = criterion(
                logits.reshape(-1, NUM_CLASSES),
                y.reshape(-1),
            ).view(y.shape)

            quality_weight = 0.75 + q.view(-1, 1)
            loss = (raw_loss * quality_weight).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * y.shape[0]
            total_items += y.shape[0]

        train_loss = total_loss / max(total_items, 1)
        val_stats = evaluate(model, val_loader, criterion, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_stats["loss"],
            "val_joint_acc": val_stats["joint_acc"],
            "val_exact_acc": val_stats["exact_acc"],
            "pred_counts": val_stats["pred_counts"],
            "true_counts": val_stats["true_counts"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_stats['loss']:.4f} | "
            f"joint_acc={val_stats['joint_acc']:.4f} | "
            f"exact={val_stats['exact_acc']:.4f}"
        )
        print("  pred:", val_stats["pred_counts"])
        print("  true:", val_stats["true_counts"])

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "state_dim": dataset.state_dim,
                    "action_dim": ACTION_DIM,
                    "num_classes": NUM_CLASSES,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "dataset": str(DATASET_PATH),
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val_loss,
                },
                MODEL_PATH,
            )

            print("  ✅ saved best")

    summary = {
        "dataset": str(DATASET_PATH),
        "model": str(MODEL_PATH),
        "items": len(dataset),
        "train_size": train_size,
        "val_size": val_size,
        "state_dim": dataset.state_dim,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nBEST")
    print(json.dumps({
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "model": str(MODEL_PATH),
    }, indent=2))


if __name__ == "__main__":
    main()
