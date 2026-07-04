#!/usr/bin/env python3
import json
import random
from pathlib import Path
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


PROJECT_DIR = Path.home() / "Documents" / "ToribashAI"

DEFAULT_MODEL_PATH = PROJECT_DIR / "models" / "parkour_active_joints_gru_v4_weight070.pt"
DEFAULT_DATASET_PATH = PROJECT_DIR / "datasets" / "ml" / "parkour_active_joints_len8.jsonl"

BATCH_SIZE = 256
SEED = 42

THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65]


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

                active_joints = obj.get("active_joints")
                action = obj.get("action")

                if active_joints is None:
                    if action is None:
                        continue
                    active_joints = [1.0 if int(v) != 0 else 0.0 for v in action]

                if states is None:
                    continue

                self.items.append((states, active_joints))

        if not self.items:
            raise RuntimeError(f"Dataset vide ou invalide: {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        states, active_joints = self.items[idx]
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(active_joints, dtype=torch.float32),
        )


class ActiveJointsGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.0 if num_layers == 1 else dropout,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


def split_fallback_dataset(path: Path):
    random.seed(SEED)

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


def evaluate_threshold(model, loader, device, threshold):
    model.eval()

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    total_joint_correct = 0
    total_joint_count = 0

    exact_correct = 0
    total_rows = 0

    pred_active_sum = 0.0
    true_active_sum = 0.0

    pred_active_distribution = Counter()
    true_active_distribution = Counter()

    per_joint_tp = torch.zeros(20)
    per_joint_fp = torch.zeros(20)
    per_joint_fn = torch.zeros(20)
    per_joint_tn = torch.zeros(20)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).float()

            correct = (preds == y)

            total_joint_correct += correct.sum().item()
            total_joint_count += correct.numel()

            exact_correct += (correct.sum(dim=1) == y.shape[1]).sum().item()
            total_rows += y.shape[0]

            pred_counts = preds.sum(dim=1).detach().cpu().tolist()
            true_counts = y.sum(dim=1).detach().cpu().tolist()

            for c in pred_counts:
                pred_active_distribution[int(c)] += 1

            for c in true_counts:
                true_active_distribution[int(c)] += 1

            pred_active_sum += sum(pred_counts)
            true_active_sum += sum(true_counts)

            batch_tp = ((preds == 1) & (y == 1)).sum().item()
            batch_fp = ((preds == 1) & (y == 0)).sum().item()
            batch_fn = ((preds == 0) & (y == 1)).sum().item()
            batch_tn = ((preds == 0) & (y == 0)).sum().item()

            tp += batch_tp
            fp += batch_fp
            fn += batch_fn
            tn += batch_tn

            per_joint_tp += ((preds == 1) & (y == 1)).sum(dim=0).detach().cpu()
            per_joint_fp += ((preds == 1) & (y == 0)).sum(dim=0).detach().cpu()
            per_joint_fn += ((preds == 0) & (y == 1)).sum(dim=0).detach().cpu()
            per_joint_tn += ((preds == 0) & (y == 0)).sum(dim=0).detach().cpu()

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-8, precision + recall)

    joint_acc = total_joint_correct / max(1, total_joint_count)
    exact_acc = exact_correct / max(1, total_rows)

    pred_active_avg = pred_active_sum / max(1, total_rows)
    true_active_avg = true_active_sum / max(1, total_rows)

    per_joint = []
    for j in range(20):
        j_tp = per_joint_tp[j].item()
        j_fp = per_joint_fp[j].item()
        j_fn = per_joint_fn[j].item()
        j_tn = per_joint_tn[j].item()

        j_precision = j_tp / max(1, j_tp + j_fp)
        j_recall = j_tp / max(1, j_tp + j_fn)
        j_f1 = (2 * j_precision * j_recall) / max(1e-8, j_precision + j_recall)

        per_joint.append(
            {
                "joint": j,
                "precision": j_precision,
                "recall": j_recall,
                "f1": j_f1,
                "tp": int(j_tp),
                "fp": int(j_fp),
                "fn": int(j_fn),
                "tn": int(j_tn),
            }
        )

    return {
        "threshold": threshold,
        "joint_acc": joint_acc,
        "exact_acc": exact_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_active_avg": pred_active_avg,
        "true_active_avg": true_active_avg,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "pred_active_distribution": sorted(pred_active_distribution.items()),
        "true_active_distribution": sorted(true_active_distribution.items()),
        "per_joint": per_joint,
    }


def main():
    model_path = DEFAULT_MODEL_PATH
    dataset_path = DEFAULT_DATASET_PATH

    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint["config"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Model: {model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Checkpoint best epoch: {checkpoint.get('best_epoch')}")
    print(f"Checkpoint best val F1: {checkpoint.get('best_val_f1')}")

    _, val_ds = split_fallback_dataset(dataset_path)

    print(f"Val sequences: {len(val_ds)}")

    loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = ActiveJointsGRU(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        output_size=config["output_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    results = []

    print()
    print("Threshold sweep:")
    print(
        "thr  | joint_acc | exact_acc | precision | recall | f1     | pred_active | true_active"
    )
    print("-" * 88)

    for threshold in THRESHOLDS:
        metrics = evaluate_threshold(model, loader, device, threshold)
        results.append(metrics)

        print(
            f"{threshold:0.2f} | "
            f"{metrics['joint_acc']:.4f}    | "
            f"{metrics['exact_acc']:.4f}    | "
            f"{metrics['precision']:.4f}    | "
            f"{metrics['recall']:.4f} | "
            f"{metrics['f1']:.4f} | "
            f"{metrics['pred_active_avg']:.2f}        | "
            f"{metrics['true_active_avg']:.2f}"
        )

    best_by_f1 = max(results, key=lambda x: x["f1"])
    best_by_active_match = min(
        results,
        key=lambda x: abs(x["pred_active_avg"] - x["true_active_avg"]),
    )

    print()
    print("Best by F1:")
    print(json.dumps(best_by_f1, indent=2))

    print()
    print("Best by active-count match:")
    print(json.dumps(best_by_active_match, indent=2))

    out_path = PROJECT_DIR / "models" / "parkour_active_joints_gru_v4_threshold_eval.json"

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": str(model_path),
                "dataset": str(dataset_path),
                "thresholds": THRESHOLDS,
                "results": results,
                "best_by_f1": best_by_f1,
                "best_by_active_match": best_by_active_match,
            },
            f,
            indent=2,
        )

    print()
    print(f"Saved eval to: {out_path}")


if __name__ == "__main__":
    main()
