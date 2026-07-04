#!/usr/bin/env python3
import json
import random
from pathlib import Path
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


PROJECT_DIR = Path.home() / "Documents" / "ToribashAI"

TRAIN_PATH = PROJECT_DIR / "datasets" / "ml" / "parkour_sequences_len8_train_mod_split.jsonl"
VAL_PATH = PROJECT_DIR / "datasets" / "ml" / "parkour_sequences_len8_val_mod_split.jsonl"

FALLBACK_PATH = PROJECT_DIR / "datasets" / "ml" / "parkour_active_joints_len8.jsonl"

MODEL_PATH = PROJECT_DIR / "models" / "parkour_active_joints_gru_v4_weight070.pt"
SUMMARY_PATH = PROJECT_DIR / "models" / "parkour_active_joints_gru_v4_weight070_summary.json"

SEQ_LEN = 8
INPUT_SIZE = 273
OUTPUT_SIZE = 20

HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.25

BATCH_SIZE = 128
EPOCHS = 12
LR = 5e-4
SEED = 42


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)


class ActiveJointsDataset(Dataset):
    def __init__(self, path: Path):
        self.items = []
        self.path = path

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                obj = json.loads(line)

                states = obj.get("states")
                if states is None:
                    states = obj.get("state_seq")

                action = obj.get("action")
                active_joints = obj.get("active_joints")

                if active_joints is None:
                    if action is None:
                        continue
                    active_joints = [1.0 if int(v) != 0 else 0.0 for v in action]

                if states is None:
                    continue

                if len(states) != SEQ_LEN:
                    continue

                if len(active_joints) != OUTPUT_SIZE:
                    continue

                self.items.append((states, active_joints))

        if not self.items:
            raise RuntimeError(f"Dataset vide ou invalide: {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        states, active_joints = self.items[idx]

        x = torch.tensor(states, dtype=torch.float32)
        y = torch.tensor(active_joints, dtype=torch.float32)

        return x, y


class ActiveJointsGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            dropout=0.0 if NUM_LAYERS == 1 else DROPOUT,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN_SIZE),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits


def find_train_val_paths():
    candidates = [
        (
            PROJECT_DIR / "datasets" / "ml" / "parkour_sequences_len8_train_mod_split.jsonl",
            PROJECT_DIR / "datasets" / "ml" / "parkour_sequences_len8_val_mod_split.jsonl",
        ),
        (
            PROJECT_DIR / "datasets" / "ml" / "train_sequences_mod_split.jsonl",
            PROJECT_DIR / "datasets" / "ml" / "val_sequences_mod_split.jsonl",
        ),
        (
            PROJECT_DIR / "datasets" / "ml" / "parkour_sequences_train_mod_split.jsonl",
            PROJECT_DIR / "datasets" / "ml" / "parkour_sequences_val_mod_split.jsonl",
        ),
    ]

    for train_path, val_path in candidates:
        if train_path.exists() and val_path.exists():
            return train_path, val_path, "mod_split"

    return FALLBACK_PATH, None, "fallback_random_split"


def split_fallback_dataset(path: Path):
    full = ActiveJointsDataset(path)

    indices = list(range(len(full)))
    random.shuffle(indices)

    val_size = max(1, int(len(indices) * 0.15))
    val_indices = set(indices[:val_size])

    train_items = []
    val_items = []

    for i, item in enumerate(full.items):
        if i in val_indices:
            val_items.append(item)
        else:
            train_items.append(item)

    train_ds = ActiveJointsDataset.__new__(ActiveJointsDataset)
    train_ds.items = train_items
    train_ds.path = path

    val_ds = ActiveJointsDataset.__new__(ActiveJointsDataset)
    val_ds.items = val_items
    val_ds.path = path

    return train_ds, val_ds


def compute_pos_weight(dataset: Dataset):
    positive = torch.zeros(OUTPUT_SIZE)
    negative = torch.zeros(OUTPUT_SIZE)

    for _, y in dataset:
        positive += y
        negative += 1.0 - y

    WEIGHT_POWER = 0.70
    pos_weight = (negative / positive.clamp(min=1.0)).pow(WEIGHT_POWER)

    return pos_weight, positive.tolist(), negative.tolist()


def metrics_from_logits(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    correct = (preds == targets).float()
    joint_acc = correct.mean().item()

    exact_acc = (correct.sum(dim=1) == OUTPUT_SIZE).float().mean().item()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()
    tn = ((preds == 0) & (targets == 0)).sum().item()

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-8, precision + recall)

    pred_active = preds.sum(dim=1).mean().item()
    true_active = targets.sum(dim=1).mean().item()

    return {
        "joint_acc": joint_acc,
        "exact_acc": exact_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_active_avg": pred_active,
        "true_active_avg": true_active,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    merged = Counter()

    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

            batch_metrics = metrics_from_logits(logits.detach(), y.detach())

            for k, v in batch_metrics.items():
                merged[k] += v

    avg_loss = total_loss / max(1, total_batches)

    avg_metrics = {
        k: v / max(1, total_batches)
        for k, v in merged.items()
        if k not in {"tp", "fp", "fn", "tn"}
    }

    for k in ["tp", "fp", "fn", "tn"]:
        avg_metrics[k] = int(merged[k])

    return avg_loss, avg_metrics


def main():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_path, val_path, split_mode = find_train_val_paths()

    print(f"Device: {device}")
    print(f"Split mode: {split_mode}")
    print(f"Train source: {train_path}")

    if split_mode == "mod_split":
        print(f"Val source: {val_path}")
        train_ds = ActiveJointsDataset(train_path)
        val_ds = ActiveJointsDataset(val_path)
    else:
        print("Aucun split par mod trouvé, fallback random split sur active dataset.")
        train_ds, val_ds = split_fallback_dataset(train_path)

    print(f"Train sequences: {len(train_ds)}")
    print(f"Val sequences: {len(val_ds)}")

    pos_weight, positive_counts, negative_counts = compute_pos_weight(train_ds)

    print("Positive counts par joint:")
    print([int(x) for x in positive_counts])

    print("Pos weight par joint:")
    print([round(float(x), 3) for x in pos_weight.tolist()])

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = ActiveJointsGRU().to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    best_val_f1 = -1.0
    best_epoch = None
    history = []

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )

        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics,
            "val": val_metrics,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_joint_acc={val_metrics['joint_acc']:.4f} | "
            f"val_exact={val_metrics['exact_acc']:.4f} | "
            f"val_precision={val_metrics['precision']:.4f} | "
            f"val_recall={val_metrics['recall']:.4f} | "
            f"val_f1={val_metrics['f1']:.4f} | "
            f"pred_active={val_metrics['pred_active_avg']:.2f} | "
            f"true_active={val_metrics['true_active_avg']:.2f}"
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "seq_len": SEQ_LEN,
                        "input_size": INPUT_SIZE,
                        "output_size": OUTPUT_SIZE,
                        "hidden_size": HIDDEN_SIZE,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "threshold": 0.5,
                    },
                    "split_mode": split_mode,
                    "train_path": str(train_path),
                    "val_path": str(val_path) if val_path else None,
                    "best_epoch": best_epoch,
                    "best_val_f1": best_val_f1,
                    "positive_counts": positive_counts,
                    "negative_counts": negative_counts,
                    "pos_weight": pos_weight.tolist(),
                },
                MODEL_PATH,
            )

            print(f"  -> saved best model: {MODEL_PATH}")

    summary = {
        "model": str(MODEL_PATH),
        "summary": str(SUMMARY_PATH),
        "split_mode": split_mode,
        "train_path": str(train_path),
        "val_path": str(val_path) if val_path else None,
        "train_sequences": len(train_ds),
        "val_sequences": len(val_ds),
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "history": history,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Done.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best val F1: {best_val_f1:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
