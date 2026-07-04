#!/usr/bin/env python3
import json
import random
from pathlib import Path
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


PROJECT_DIR = Path.home() / "Documents" / "ToribashAI"

DATASET_PATH = PROJECT_DIR / "datasets" / "ml" / "parkour_active_values_len8.jsonl"
MODEL_PATH = PROJECT_DIR / "models" / "parkour_active_values_gru_v1.pt"
SUMMARY_PATH = PROJECT_DIR / "models" / "parkour_active_values_gru_v1_eval.json"

SEQ_LEN = 8
STATE_SIZE = 273
NUM_JOINTS = 20
INPUT_SIZE = STATE_SIZE + NUM_JOINTS

HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.25
NUM_CLASSES = 4

BATCH_SIZE = 512
SEED = 42


class ActiveValueDataset(Dataset):
    def __init__(self, path):
        self.items = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                obj = json.loads(line)

                states = obj["states"]
                joint_id = int(obj["joint_id"])
                target_class = int(obj["target_class"])

                self.items.append((states, joint_id, target_class))

        if not self.items:
            raise RuntimeError("Dataset vide")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        states, joint_id, target_class = self.items[idx]

        joint_onehot = torch.zeros(NUM_JOINTS)
        joint_onehot[joint_id] = 1.0

        states = torch.tensor(states, dtype=torch.float32)
        joint_seq = joint_onehot.unsqueeze(0).repeat(SEQ_LEN, 1)

        x = torch.cat([states, joint_seq], dim=1)
        y = torch.tensor(target_class, dtype=torch.long)
        joint = torch.tensor(joint_id, dtype=torch.long)

        return x, y, joint


class ActiveValueGRU(nn.Module):
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


def split_dataset(dataset):
    random.seed(SEED)

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    val_size = int(len(indices) * 0.15)
    val_indices = set(indices[:val_size])

    train_items = []
    val_items = []

    for i, item in enumerate(dataset.items):
        if i in val_indices:
            val_items.append(item)
        else:
            train_items.append(item)

    train_ds = ActiveValueDataset.__new__(ActiveValueDataset)
    train_ds.items = train_items

    val_ds = ActiveValueDataset.__new__(ActiveValueDataset)
    val_ds.items = val_items

    return train_ds, val_ds


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Model: {MODEL_PATH}")

    full_ds = ActiveValueDataset(DATASET_PATH)
    _train_ds, val_ds = split_dataset(full_ds)

    print(f"Val examples: {len(val_ds)}")

    loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = ActiveValueGRU().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total = 0
    correct = 0

    pred_counter = Counter()
    true_counter = Counter()

    joint_total = Counter()
    joint_correct = Counter()

    # confusion[true][pred]
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)

    with torch.no_grad():
        for x, y, joint in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            total += y.shape[0]
            correct += (preds == y).sum().item()

            for t, p, j in zip(
                y.detach().cpu().tolist(),
                preds.detach().cpu().tolist(),
                joint.detach().cpu().tolist(),
            ):
                true_value = int(t) + 1
                pred_value = int(p) + 1

                true_counter[true_value] += 1
                pred_counter[pred_value] += 1

                joint_total[int(j)] += 1
                if int(t) == int(p):
                    joint_correct[int(j)] += 1

                confusion[int(t), int(p)] += 1

    accuracy = correct / max(1, total)

    print()
    print("Active Values Eval")
    print("------------------")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Total:    {total}")

    print()
    print("True value distribution:")
    print(sorted(true_counter.items()))

    print()
    print("Predicted value distribution:")
    print(sorted(pred_counter.items()))

    print()
    print("Confusion matrix rows=true, cols=pred")
    print("       pred1  pred2  pred3  pred4")
    for i in range(NUM_CLASSES):
        row = confusion[i].tolist()
        print(
            f"true{i+1} "
            f"{row[0]:7d} {row[1]:7d} {row[2]:7d} {row[3]:7d}"
        )

    print()
    print("Accuracy by true value:")
    per_value = []
    for i in range(NUM_CLASSES):
        row_total = confusion[i].sum().item()
        row_correct = confusion[i, i].item()
        row_acc = row_correct / max(1, row_total)

        per_value.append(
            {
                "value": i + 1,
                "total": int(row_total),
                "correct": int(row_correct),
                "accuracy": row_acc,
            }
        )

        print(
            f"value {i+1}: "
            f"acc={row_acc:.4f} "
            f"correct={row_correct} "
            f"total={row_total}"
        )

    print()
    print("Accuracy by joint:")
    per_joint = []
    for joint_id in range(NUM_JOINTS):
        total_j = joint_total[joint_id]
        correct_j = joint_correct[joint_id]
        acc_j = correct_j / max(1, total_j)

        per_joint.append(
            {
                "joint": joint_id,
                "total": int(total_j),
                "correct": int(correct_j),
                "accuracy": acc_j,
            }
        )

        print(
            f"joint {joint_id:02d}: "
            f"acc={acc_j:.4f} "
            f"correct={correct_j} "
            f"total={total_j}"
        )

    summary = {
        "model": str(MODEL_PATH),
        "dataset": str(DATASET_PATH),
        "accuracy": accuracy,
        "total": total,
        "true_distribution": sorted(true_counter.items()),
        "pred_distribution": sorted(pred_counter.items()),
        "confusion": confusion.tolist(),
        "per_value": per_value,
        "per_joint": per_joint,
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Saved eval to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
