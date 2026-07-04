#!/usr/bin/env python3
import json
import random
from pathlib import Path
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


PROJECT_DIR = Path.home() / "Documents" / "ToribashAI"

MODEL_PATH = PROJECT_DIR / "models" / "parkour_active_joints_gru_v4_weight070.pt"
DATASET_PATH = PROJECT_DIR / "datasets" / "ml" / "parkour_active_joints_len8.jsonl"
OUT_PATH = PROJECT_DIR / "models" / "parkour_active_joints_topk_oracle_eval.json"

BATCH_SIZE = 256
SEED = 42


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
                active_count = obj.get("active_count")

                if active_joints is None:
                    if action is None:
                        continue
                    active_joints = [1.0 if int(v) != 0 else 0.0 for v in action]

                if active_count is None:
                    active_count = sum(int(v) for v in active_joints)

                if states is None:
                    continue

                if len(active_joints) != 20:
                    continue

                active_count = int(active_count)

                if active_count < 0 or active_count > 20:
                    continue

                self.items.append((states, active_joints, active_count))

        if not self.items:
            raise RuntimeError(f"Dataset vide ou invalide: {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        states, active_joints, active_count = self.items[idx]

        x = torch.tensor(states, dtype=torch.float32)
        y = torch.tensor(active_joints, dtype=torch.float32)
        k = torch.tensor(active_count, dtype=torch.long)

        return x, y, k


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


def topk_from_probs(probs, k_values):
    batch_size, num_joints = probs.shape

    preds = torch.zeros_like(probs)

    for i in range(batch_size):
        k = int(k_values[i].item())

        if k <= 0:
            continue

        if k >= num_joints:
            preds[i, :] = 1.0
            continue

        top_indices = torch.topk(probs[i], k=k).indices
        preds[i, top_indices] = 1.0

    return preds


def evaluate_topk_oracle(model, loader, device):
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

    pred_active_dist = Counter()
    true_active_dist = Counter()

    per_k = {}

    per_joint_tp = torch.zeros(20)
    per_joint_fp = torch.zeros(20)
    per_joint_fn = torch.zeros(20)
    per_joint_tn = torch.zeros(20)

    with torch.no_grad():
        for x, y, k in loader:
            x = x.to(device)
            y = y.to(device)
            k = k.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)

            preds = topk_from_probs(probs, k)

            correct = preds == y

            total_joint_correct += correct.sum().item()
            total_joint_count += correct.numel()

            exact_correct += (correct.sum(dim=1) == y.shape[1]).sum().item()
            total_rows += y.shape[0]

            pred_counts = preds.sum(dim=1).detach().cpu().tolist()
            true_counts = y.sum(dim=1).detach().cpu().tolist()

            for c in pred_counts:
                pred_active_dist[int(c)] += 1

            for c in true_counts:
                true_active_dist[int(c)] += 1

            pred_active_sum += sum(pred_counts)
            true_active_sum += sum(true_counts)

            batch_tp_tensor = ((preds == 1) & (y == 1))
            batch_fp_tensor = ((preds == 1) & (y == 0))
            batch_fn_tensor = ((preds == 0) & (y == 1))
            batch_tn_tensor = ((preds == 0) & (y == 0))

            tp += batch_tp_tensor.sum().item()
            fp += batch_fp_tensor.sum().item()
            fn += batch_fn_tensor.sum().item()
            tn += batch_tn_tensor.sum().item()

            per_joint_tp += batch_tp_tensor.sum(dim=0).detach().cpu()
            per_joint_fp += batch_fp_tensor.sum(dim=0).detach().cpu()
            per_joint_fn += batch_fn_tensor.sum(dim=0).detach().cpu()
            per_joint_tn += batch_tn_tensor.sum(dim=0).detach().cpu()

            for i in range(y.shape[0]):
                kk = int(k[i].item())

                if kk not in per_k:
                    per_k[kk] = {
                        "rows": 0,
                        "exact": 0,
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "tn": 0,
                    }

                row_correct = correct[i]
                row_tp = batch_tp_tensor[i].sum().item()
                row_fp = batch_fp_tensor[i].sum().item()
                row_fn = batch_fn_tensor[i].sum().item()
                row_tn = batch_tn_tensor[i].sum().item()

                per_k[kk]["rows"] += 1
                per_k[kk]["exact"] += int(row_correct.sum().item() == 20)
                per_k[kk]["tp"] += int(row_tp)
                per_k[kk]["fp"] += int(row_fp)
                per_k[kk]["fn"] += int(row_fn)
                per_k[kk]["tn"] += int(row_tn)

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

    per_k_rows = []
    for kk in sorted(per_k.keys()):
        item = per_k[kk]

        kk_tp = item["tp"]
        kk_fp = item["fp"]
        kk_fn = item["fn"]
        kk_tn = item["tn"]

        kk_precision = kk_tp / max(1, kk_tp + kk_fp)
        kk_recall = kk_tp / max(1, kk_tp + kk_fn)
        kk_f1 = (2 * kk_precision * kk_recall) / max(1e-8, kk_precision + kk_recall)
        kk_exact = item["exact"] / max(1, item["rows"])

        per_k_rows.append(
            {
                "active_count": kk,
                "rows": item["rows"],
                "exact_acc": kk_exact,
                "precision": kk_precision,
                "recall": kk_recall,
                "f1": kk_f1,
                "tp": kk_tp,
                "fp": kk_fp,
                "fn": kk_fn,
                "tn": kk_tn,
            }
        )

    return {
        "mode": "topk_oracle_true_active_count",
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
        "pred_active_distribution": sorted(pred_active_dist.items()),
        "true_active_distribution": sorted(true_active_dist.items()),
        "per_k": per_k_rows,
        "per_joint": per_joint,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    config = checkpoint["config"]

    print(f"Device: {device}")
    print(f"Model: {MODEL_PATH}")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Checkpoint best epoch: {checkpoint.get('best_epoch')}")
    print(f"Checkpoint best val F1: {checkpoint.get('best_val_f1')}")

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

    metrics = evaluate_topk_oracle(model, loader, device)

    print()
    print("Top-K Oracle Eval")
    print("-----------------")
    print(f"joint_acc:       {metrics['joint_acc']:.4f}")
    print(f"exact_acc:       {metrics['exact_acc']:.4f}")
    print(f"precision:       {metrics['precision']:.4f}")
    print(f"recall:          {metrics['recall']:.4f}")
    print(f"f1:              {metrics['f1']:.4f}")
    print(f"pred_active_avg: {metrics['pred_active_avg']:.2f}")
    print(f"true_active_avg: {metrics['true_active_avg']:.2f}")

    print()
    print("Pred active distribution:")
    print(metrics["pred_active_distribution"])

    print()
    print("True active distribution:")
    print(metrics["true_active_distribution"])

    print()
    print("Per-K summary:")
    for row in metrics["per_k"]:
        if row["rows"] < 5:
            continue

        print(
            f"K={row['active_count']:02d} | "
            f"rows={row['rows']:4d} | "
            f"exact={row['exact_acc']:.4f} | "
            f"precision={row['precision']:.4f} | "
            f"recall={row['recall']:.4f} | "
            f"f1={row['f1']:.4f}"
        )

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print()
    print(f"Saved eval to: {OUT_PATH}")


if __name__ == "__main__":
    main()
