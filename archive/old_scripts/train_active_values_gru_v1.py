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
SUMMARY_PATH = PROJECT_DIR / "models" / "parkour_active_values_gru_v1_summary.json"

SEQ_LEN = 8
STATE_SIZE = 273
NUM_JOINTS = 20

INPUT_SIZE = STATE_SIZE + NUM_JOINTS

HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.25

NUM_CLASSES = 4

BATCH_SIZE = 256
EPOCHS = 12
LR = 5e-4
SEED = 42


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


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

                self.items.append(
                    (
                        states,
                        joint_id,
                        target_class,
                    )
                )

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

        x = torch.cat(
            [states, joint_seq],
            dim=1
        )

        y = torch.tensor(target_class, dtype=torch.long)

        return x, y


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


def compute_class_weights(dataset):
    counter = Counter()

    for _, _, cls in dataset.items:
        counter[cls] += 1

    total = sum(counter.values())

    weights = []

    for cls in range(NUM_CLASSES):
        count = counter[cls]
        weight = total / (NUM_CLASSES * count)
        weights.append(weight)

    weights = torch.tensor(weights, dtype=torch.float32)

    weights = weights.pow(0.35)

    return weights, counter


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0

    total_correct = 0
    total_samples = 0

    pred_counter = Counter()
    true_counter = Counter()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(logits, y)

            preds = logits.argmax(dim=1)

            total_loss += loss.item()

            total_correct += (preds == y).sum().item()
            total_samples += y.shape[0]

            for v in preds.cpu().tolist():
                pred_counter[v + 1] += 1

            for v in y.cpu().tolist():
                true_counter[v + 1] += 1

    return {
        "loss": total_loss / len(loader),
        "acc": total_correct / total_samples,
        "pred_dist": sorted(pred_counter.items()),
        "true_dist": sorted(true_counter.items()),
    }


def main():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    full_ds = ActiveValueDataset(DATASET_PATH)

    train_ds, val_ds = split_dataset(full_ds)

    print(f"Train examples: {len(train_ds)}")
    print(f"Val examples: {len(val_ds)}")

    class_weights, class_counts = compute_class_weights(train_ds)

    print("Class counts:")
    print(sorted((k + 1, v) for k, v in class_counts.items()))

    print("Class weights:")
    print([round(float(v), 3) for v in class_weights.tolist()])

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

    model = ActiveValueGRU().to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )

    best_acc = 0.0

    history = []

    for epoch in range(1, EPOCHS + 1):

        model.train()

        train_loss = 0.0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **val_metrics
            }
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["acc"] > best_acc:
            best_acc = val_metrics["acc"]

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_acc": best_acc,
                },
                MODEL_PATH,
            )

            print(f"  -> saved best model: {MODEL_PATH}")

    summary = {
        "best_acc": best_acc,
        "history": history,
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)

    print()
    print("Done.")
    print(f"Best accuracy: {best_acc:.4f}")
    print(f"Model: {MODEL_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
