#!/usr/bin/env python3
import json
import random
import math
from pathlib import Path
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


PROJECT_DIR = Path.home() / "Documents" / "ToribashAI"

MODEL_PATH = PROJECT_DIR / "models" / "parkour_active_joints_gru_v4_weight070.pt"
DATASET_PATH = PROJECT_DIR / "datasets" / "ml" / "parkour_active_joints_len8.jsonl"
OUT_PATH = PROJECT_DIR / "models" / "parkour_active_joints_dynamic_k_eval.json"

BATCH_SIZE = 256
SEED = 42


class ActiveJointsDataset(Dataset):
    def __init__(self, path: Path):
        self.items = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                obj = json.loads(line)

                states = obj.get("states") or obj.get("state_seq")
                active_joints = obj.get("active_joints")
                action = obj.get("action")

                if active_joints is None:
                    if action is None:
                        continue
                    active_joints = [1.0 if int(v) != 0 else 0.0 for v in action]

                if states is None or len(active_joints) != 20:
                    continue

                active_count = int(obj.get("active_count", sum(int(v) for v in active_joints)))

                self.items.append((states, active_joints, active_count))

        if not self.items:
            raise RuntimeError(f"Dataset vide ou invalide: {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        states, active_joints, active_count = self.items[idx]

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(active_joints, dtype=torch.float32),
            torch.tensor(active_count, dtype=torch.long),
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
        return self.head(out[:, -1, :])


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

    val_ds = ActiveJointsDataset.__new__(ActiveJointsDataset)
    val_ds.items = val_items

    return train_ds, val_ds


def build_prediction_from_k(probs, k_values):
    preds = torch.zeros_like(probs)

    batch_size, num_joints = probs.shape

    for i in range(batch_size):
        k = int(k_values[i])

        k = max(1, min(num_joints, k))

        top_indices = torch.topk(probs[i], k=k).indices
        preds[i, top_indices] = 1.0

    return preds


def compute_metrics(preds, y, true_k):
    tp = ((preds == 1) & (y == 1)).sum().item()
    fp = ((preds == 1) & (y == 0)).sum().item()
    fn = ((preds == 0) & (y == 1)).sum().item()
    tn = ((preds == 0) & (y == 0)).sum().item()

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-8, precision + recall)

    correct = preds == y

    joint_acc = correct.float().mean().item()
    exact_acc = (correct.sum(dim=1) == y.shape[1]).float().mean().item()

    pred_k = preds.sum(dim=1)

    pred_avg = pred_k.float().mean().item()
    true_avg = true_k.float().mean().item()

    k_mae = (pred_k.float() - true_k.float()).abs().mean().item()
    k_within_1 = ((pred_k.float() - true_k.float()).abs() <= 1).float().mean().item()
    k_within_2 = ((pred_k.float() - true_k.float()).abs() <= 2).float().mean().item()

    return {
        "joint_acc": joint_acc,
        "exact_acc": exact_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pred_active_avg": pred_avg,
        "true_active_avg": true_avg,
        "k_mae": k_mae,
        "k_within_1": k_within_1,
        "k_within_2": k_within_2,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def k_from_strategy(probs, strategy):
    batch_size, num_joints = probs.shape

    if strategy["type"] == "threshold":
        threshold = strategy["threshold"]
        k = (probs >= threshold).sum(dim=1)
        k = k.clamp(min=1, max=num_joints)
        return k.long()

    if strategy["type"] == "sum_round":
        scale = strategy["scale"]
        k = torch.round(probs.sum(dim=1) * scale)
        k = k.clamp(min=1, max=num_joints)
        return k.long()

    if strategy["type"] == "sum_ceil":
        scale = strategy["scale"]
        k = torch.ceil(probs.sum(dim=1) * scale)
        k = k.clamp(min=1, max=num_joints)
        return k.long()

    if strategy["type"] == "sum_floor":
        scale = strategy["scale"]
        k = torch.floor(probs.sum(dim=1) * scale)
        k = k.clamp(min=1, max=num_joints)
        return k.long()

    if strategy["type"] == "fixed":
        k = torch.full((batch_size,), strategy["k"], device=probs.device)
        return k.long()

    raise ValueError(f"Unknown strategy: {strategy}")


def evaluate_strategy(model, loader, device, strategy):
    model.eval()

    merged = Counter()
    batches = 0

    pred_dist = Counter()
    true_dist = Counter()

    with torch.no_grad():
        for x, y, true_k in loader:
            x = x.to(device)
            y = y.to(device)
            true_k = true_k.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)

            pred_k = k_from_strategy(probs, strategy)
            preds = build_prediction_from_k(probs, pred_k)

            metrics = compute_metrics(preds, y, true_k)

            for key, value in metrics.items():
                merged[key] += value

            batches += 1

            for v in pred_k.detach().cpu().tolist():
                pred_dist[int(v)] += 1

            for v in true_k.detach().cpu().tolist():
                true_dist[int(v)] += 1

    avg = {
        key: value / max(1, batches)
        for key, value in merged.items()
        if key not in {"tp", "fp", "fn", "tn"}
    }

    for key in ["tp", "fp", "fn", "tn"]:
        avg[key] = int(merged[key])

    avg["strategy"] = strategy
    avg["pred_active_distribution"] = sorted(pred_dist.items())
    avg["true_active_distribution"] = sorted(true_dist.items())

    return avg


def strategy_name(strategy):
    if strategy["type"] == "threshold":
        return f"thr>{strategy['threshold']:.2f}"
    if strategy["type"] in {"sum_round", "sum_ceil", "sum_floor"}:
        return f"{strategy['type']}*{strategy['scale']:.2f}"
    if strategy["type"] == "fixed":
        return f"fixed{strategy['k']}"
    return str(strategy)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    config = checkpoint["config"]

    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")

    _, val_ds = split_fallback_dataset(DATASET_PATH)

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

    strategies = []

    for threshold in [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
        strategies.append({"type": "threshold", "threshold": threshold})

    for scale in [0.55, 0.65, 0.75, 0.85, 0.95, 1.00, 1.10, 1.20]:
        strategies.append({"type": "sum_round", "scale": scale})
        strategies.append({"type": "sum_ceil", "scale": scale})
        strategies.append({"type": "sum_floor", "scale": scale})

    for k in [1, 2, 3, 4, 5, 6, 7, 8]:
        strategies.append({"type": "fixed", "k": k})

    results = []

    print()
    print("Dynamic K eval:")
    print("strategy          | f1     | precision | recall | joint_acc | exact | predK | trueK | k_mae | k±1")
    print("-" * 100)

    for strategy in strategies:
        metrics = evaluate_strategy(model, loader, device, strategy)
        results.append(metrics)

        print(
            f"{strategy_name(strategy):16s} | "
            f"{metrics['f1']:.4f} | "
            f"{metrics['precision']:.4f}    | "
            f"{metrics['recall']:.4f} | "
            f"{metrics['joint_acc']:.4f}    | "
            f"{metrics['exact_acc']:.4f} | "
            f"{metrics['pred_active_avg']:.2f}  | "
            f"{metrics['true_active_avg']:.2f}  | "
            f"{metrics['k_mae']:.2f} | "
            f"{metrics['k_within_1']:.4f}"
        )

    best_by_f1 = max(results, key=lambda x: x["f1"])
    best_by_k = min(results, key=lambda x: x["k_mae"])
    best_combo = max(
        results,
        key=lambda x: x["f1"] - 0.04 * abs(x["pred_active_avg"] - x["true_active_avg"]),
    )

    print()
    print("Best by F1:")
    print(json.dumps(best_by_f1, indent=2))

    print()
    print("Best by K MAE:")
    print(json.dumps(best_by_k, indent=2))

    print()
    print("Best combo:")
    print(json.dumps(best_combo, indent=2))

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model": str(MODEL_PATH),
                "dataset": str(DATASET_PATH),
                "results": results,
                "best_by_f1": best_by_f1,
                "best_by_k": best_by_k,
                "best_combo": best_combo,
            },
            f,
            indent=2,
        )

    print()
    print(f"Saved eval to: {OUT_PATH}")


if __name__ == "__main__":
    main()
