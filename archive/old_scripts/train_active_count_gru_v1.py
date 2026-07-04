#!/usr/bin/env python3
import json
import random
from pathlib import Path
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


PROJECT_DIR = Path.home() / "Documents" / "ToribashAI"

DATASET_PATH = PROJECT_DIR / "datasets" / "ml" / "parkour_active_joints_len8.jsonl"

MODEL_PATH = PROJECT_DIR / "models" / "parkour_active_count_gru_v1.pt"
SUMMARY_PATH = PROJECT_DIR / "models" / "parkour_active_count_gru_v1_summary.json"

SEQ_LEN = 8
INPUT_SIZE = 273
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.25

NUM_CLASSES = 20  # active_count 1..20 -> classes 0..19

BATCH_SIZE = 128
EPOCHS = 14
LR = 5e-4
SEED = 42


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


class ActiveCountDataset(Dataset):
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

                active_count = obj.get("active_count")

                if active_count is None:
                    active_joints = obj.get("active_joints")
                    action = obj.get("action")

                    if active_joints is not None:
                        active_count = sum(int(v) for v in active_joints)
                    elif action is not None:
                        active_count = sum(1 for v in action if int(v) != 0)
                    else:
                        continue

                if states is None:
                    continue

                if len(states) != SEQ_LEN:
                    continue

                active_count = int(active_count)

                if active_count < 1 or active_count > 20:
                    continue

                label = active_count - 1
                self.items.append((states, label, active_count))

        if not self.items:
            raise RuntimeError(f"Dataset vide ou invalide: {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        states, label, active_count = self.items[idx]

        x = torch.tensor(states, dtype=torch.float32)
        y = torch.tensor(label, dtype=torch.long)

        return x, y, active_count


class ActiveCountGRU(nn.Module):
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
            nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


def split_dataset(path: Path):
    full = ActiveCountDataset(path)

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

    train_ds = ActiveCountDataset.__new__(ActiveCountDataset)
    train_ds.items = train_items
    train_ds.path = path

    val_ds = ActiveCountDataset.__new__(ActiveCountDataset)
    val_ds.items = val_items
    val_ds.path = path

    return train_ds, val_ds


def compute_class_weights(dataset):
    counts = Counter()

    for _, label, _ in dataset.items:
        counts[label] += 1

    total = sum(counts.values())
    weights = []

    for cls in range(NUM_CLASSES):
        count = counts.get(cls, 0)
        if count == 0:
            weights.append(0.0)
        else:
            weights.append(total / (NUM_CLASSES * count))

    weights = torch.tensor(weights, dtype=torch.float32)

    # adoucissement pour éviter de sur-prédire les classes rares
    weights = weights.pow(0.45)

    return weights, counts


def metrics_from_logits(logits, y):
    preds = logits.argmax(dim=1)

    pred_counts = preds + 1
    true_counts = y + 1

    acc = (preds == y).float().mean().item()
    mae = (pred_counts.float() - true_counts.float()).abs().mean().item()

    pred_avg = pred_counts.float().mean().item()
    true_avg = true_counts.float().mean().item()

    within_1 = ((pred_counts - true_counts).abs() <= 1).float().mean().item()
    within_2 = ((pred_counts - true_counts).abs() <= 2).float().mean().item()

    return {
        "acc": acc,
        "mae": mae,
        "pred_avg": pred_avg,
        "true_avg": true_avg,
        "within_1": within_1,
        "within_2": within_2,
    }


def run_epoch(model, loader, criterion, optimizer, device, train):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_batches = 0

    merged = Counter()

    with torch.set_grad_enabled(train):
        for x, y, _active_count in loader:
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
    avg_metrics = {k: v / max(1, total_batches) for k, v in merged.items()}

    return avg_loss, avg_metrics


def evaluate_distribution(model, loader, device):
    model.eval()

    pred_dist = Counter()
    true_dist = Counter()

    with torch.no_grad():
        for x, y, _active_count in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            for v in (preds + 1).detach().cpu().tolist():
                pred_dist[int(v)] += 1

            for v in (y + 1).detach().cpu().tolist():
                true_dist[int(v)] += 1

    return sorted(pred_dist.items()), sorted(true_dist.items())


def main():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Dataset: {DATASET_PATH}")

    train_ds, val_ds = split_dataset(DATASET_PATH)

    print(f"Train sequences: {len(train_ds)}")
    print(f"Val sequences: {len(val_ds)}")

    class_weights, class_counts = compute_class_weights(train_ds)

    print("Train active_count distribution:")
    print(sorted((k + 1, v) for k, v in class_counts.items()))

    print("Class weights:")
    print([round(float(v), 3) for v in class_weights.tolist()])

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

    model = ActiveCountGRU().to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_score = -999.0
    best_epoch = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )

        val_loss, val_metrics = run_epoch(
            model, val_loader, criterion, optimizer, device, train=False
        )

        # Score mixte : on veut une bonne approximation du nombre de joints,
        # pas seulement une accuracy brute.
        score = val_metrics["within_1"] - (val_metrics["mae"] * 0.05)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train": train_metrics,
            "val": val_metrics,
            "score": score,
        }

        history.append(row)

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_metrics['acc']:.4f} | "
            f"val_mae={val_metrics['mae']:.3f} | "
            f"within1={val_metrics['within_1']:.4f} | "
            f"within2={val_metrics['within_2']:.4f} | "
            f"pred_avg={val_metrics['pred_avg']:.2f} | "
            f"true_avg={val_metrics['true_avg']:.2f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {
                        "seq_len": SEQ_LEN,
                        "input_size": INPUT_SIZE,
                        "hidden_size": HIDDEN_SIZE,
                        "num_layers": NUM_LAYERS,
                        "dropout": DROPOUT,
                        "num_classes": NUM_CLASSES,
                    },
                    "dataset_path": str(DATASET_PATH),
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    "class_weights": class_weights.tolist(),
                },
                MODEL_PATH,
            )

            print(f"  -> saved best model: {MODEL_PATH}")

    pred_dist, true_dist = evaluate_distribution(model, val_loader, device)

    summary = {
        "model": str(MODEL_PATH),
        "summary": str(SUMMARY_PATH),
        "dataset": str(DATASET_PATH),
        "train_sequences": len(train_ds),
        "val_sequences": len(val_ds),
        "best_epoch": best_epoch,
        "best_score": best_score,
        "history": history,
        "final_pred_distribution": pred_dist,
        "final_true_distribution": true_dist,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Done.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best score: {best_score:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")

    print()
    print("Final predicted active_count distribution:")
    print(pred_dist)

    print()
    print("True active_count distribution:")
    print(true_dist)


if __name__ == "__main__":
    main()
