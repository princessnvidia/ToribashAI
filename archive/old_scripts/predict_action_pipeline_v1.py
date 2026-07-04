#!/usr/bin/env python3
import json
import random
from pathlib import Path
from collections import Counter

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


PROJECT_DIR = Path.home() / "Documents" / "ToribashAI"

SEQUENCES_PATH = PROJECT_DIR / "datasets" / "ml" / "parkour_active_joints_len8.jsonl"

ACTIVE_JOINTS_MODEL_PATH = PROJECT_DIR / "models" / "parkour_active_joints_gru_v4_weight070.pt"
ACTIVE_VALUES_MODEL_PATH = PROJECT_DIR / "models" / "parkour_active_values_gru_v1.pt"

OUT_PATH = PROJECT_DIR / "models" / "pipeline_v1_predictions_sample.json"

SEQ_LEN = 8
STATE_SIZE = 273
NUM_JOINTS = 20
VALUE_INPUT_SIZE = STATE_SIZE + NUM_JOINTS

ACTIVE_THRESHOLD = 0.45

BATCH_SIZE = 128
SEED = 42
MAX_SAVED_EXAMPLES = 80


class SequenceDataset(Dataset):
    def __init__(self, path: Path):
        self.items = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                obj = json.loads(line)

                states = obj.get("states") or obj.get("state_seq")
                action = obj.get("action")

                if action is None:
                    active_joints = obj.get("active_joints")
                    if active_joints is None:
                        continue
                    action = [0 for _ in range(NUM_JOINTS)]
                else:
                    action = [int(v) for v in action]

                if states is None:
                    continue

                if len(states) != SEQ_LEN:
                    continue

                if len(action) != NUM_JOINTS:
                    continue

                self.items.append((states, action))

        if not self.items:
            raise RuntimeError(f"Dataset vide ou invalide: {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        states, action = self.items[idx]

        x = torch.tensor(states, dtype=torch.float32)
        y = torch.tensor(action, dtype=torch.long)

        return x, y


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


class ActiveValueGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=VALUE_INPUT_SIZE,
            hidden_size=128,
            num_layers=1,
            dropout=0.0,
            batch_first=True,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Dropout(0.25),
            nn.Linear(128, 4),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.head(last)


def split_val_dataset(path: Path):
    random.seed(SEED)

    full = SequenceDataset(path)

    indices = list(range(len(full)))
    random.shuffle(indices)

    val_size = max(1, int(len(indices) * 0.15))
    val_indices = set(indices[:val_size])

    val_items = []

    for i, item in enumerate(full.items):
        if i in val_indices:
            val_items.append(item)

    val_ds = SequenceDataset.__new__(SequenceDataset)
    val_ds.items = val_items

    return val_ds


def make_value_input(states_batch, joint_ids):
    batch_size = states_batch.shape[0]

    joint_onehot = torch.zeros(
        batch_size,
        NUM_JOINTS,
        dtype=states_batch.dtype,
        device=states_batch.device,
    )

    joint_onehot[torch.arange(batch_size, device=states_batch.device), joint_ids] = 1.0

    joint_seq = joint_onehot.unsqueeze(1).repeat(1, SEQ_LEN, 1)

    value_input = torch.cat([states_batch, joint_seq], dim=2)

    return value_input


def predict_pipeline(states, active_model, value_model, device):
    states = states.to(device)

    active_logits = active_model(states)
    active_probs = torch.sigmoid(active_logits)

    active_mask = active_probs >= ACTIVE_THRESHOLD

    batch_size = states.shape[0]

    pred_actions = torch.zeros(
        batch_size,
        NUM_JOINTS,
        dtype=torch.long,
        device=device,
    )

    for joint_id in range(NUM_JOINTS):
        row_mask = active_mask[:, joint_id]

        if row_mask.sum().item() == 0:
            continue

        selected_states = states[row_mask]
        joint_ids = torch.full(
            (selected_states.shape[0],),
            joint_id,
            dtype=torch.long,
            device=device,
        )

        value_input = make_value_input(selected_states, joint_ids)
        value_logits = value_model(value_input)
        value_classes = value_logits.argmax(dim=1)

        pred_values = value_classes + 1

        pred_actions[row_mask, joint_id] = pred_values

    return pred_actions, active_probs


def compute_metrics(pred_actions, true_actions):
    total_values = true_actions.numel()

    joint_acc = (pred_actions == true_actions).float().mean().item()

    exact_acc = (
        (pred_actions == true_actions).sum(dim=1) == NUM_JOINTS
    ).float().mean().item()

    pred_active = pred_actions != 0
    true_active = true_actions != 0

    active_correct = pred_active == true_active

    active_joint_acc = active_correct.float().mean().item()

    tp = ((pred_active == 1) & (true_active == 1)).sum().item()
    fp = ((pred_active == 1) & (true_active == 0)).sum().item()
    fn = ((pred_active == 0) & (true_active == 1)).sum().item()
    tn = ((pred_active == 0) & (true_active == 0)).sum().item()

    active_precision = tp / max(1, tp + fp)
    active_recall = tp / max(1, tp + fn)
    active_f1 = (2 * active_precision * active_recall) / max(
        1e-8,
        active_precision + active_recall,
    )

    pred_active_avg = pred_active.sum(dim=1).float().mean().item()
    true_active_avg = true_active.sum(dim=1).float().mean().item()

    value_positions = true_active

    if value_positions.sum().item() > 0:
        value_acc_on_true_active = (
            pred_actions[value_positions] == true_actions[value_positions]
        ).float().mean().item()
    else:
        value_acc_on_true_active = 0.0

    pred_counter = Counter(pred_actions.detach().cpu().flatten().tolist())
    true_counter = Counter(true_actions.detach().cpu().flatten().tolist())

    return {
        "joint_acc": joint_acc,
        "exact_acc": exact_acc,
        "active_joint_acc": active_joint_acc,
        "active_precision": active_precision,
        "active_recall": active_recall,
        "active_f1": active_f1,
        "pred_active_avg": pred_active_avg,
        "true_active_avg": true_active_avg,
        "value_acc_on_true_active": value_acc_on_true_active,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "pred_value_distribution": sorted((int(k), int(v)) for k, v in pred_counter.items()),
        "true_value_distribution": sorted((int(k), int(v)) for k, v in true_counter.items()),
        "total_values": int(total_values),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Sequences: {SEQUENCES_PATH}")
    print(f"Active joints model: {ACTIVE_JOINTS_MODEL_PATH}")
    print(f"Active values model: {ACTIVE_VALUES_MODEL_PATH}")
    print(f"Active threshold: {ACTIVE_THRESHOLD}")

    active_ckpt = torch.load(ACTIVE_JOINTS_MODEL_PATH, map_location="cpu")
    active_config = active_ckpt["config"]

    active_model = ActiveJointsGRU(
        input_size=active_config["input_size"],
        hidden_size=active_config["hidden_size"],
        output_size=active_config["output_size"],
        num_layers=active_config["num_layers"],
        dropout=active_config["dropout"],
    ).to(device)

    active_model.load_state_dict(active_ckpt["model_state_dict"])
    active_model.eval()

    value_ckpt = torch.load(ACTIVE_VALUES_MODEL_PATH, map_location="cpu")

    value_model = ActiveValueGRU().to(device)
    value_model.load_state_dict(value_ckpt["model_state_dict"])
    value_model.eval()

    val_ds = split_val_dataset(SEQUENCES_PATH)

    print(f"Val sequences: {len(val_ds)}")

    loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    merged = Counter()
    scalar_sums = Counter()
    batches = 0

    samples = []

    with torch.no_grad():
        for states, true_actions in loader:
            true_actions = true_actions.to(device)

            pred_actions, active_probs = predict_pipeline(
                states,
                active_model,
                value_model,
                device,
            )

            metrics = compute_metrics(pred_actions, true_actions)

            for key in [
                "tp",
                "fp",
                "fn",
                "tn",
                "total_values",
            ]:
                merged[key] += metrics[key]

            for key in [
                "joint_acc",
                "exact_acc",
                "active_joint_acc",
                "active_precision",
                "active_recall",
                "active_f1",
                "pred_active_avg",
                "true_active_avg",
                "value_acc_on_true_active",
            ]:
                scalar_sums[key] += metrics[key]

            batches += 1

            pred_cpu = pred_actions.detach().cpu().tolist()
            true_cpu = true_actions.detach().cpu().tolist()
            probs_cpu = active_probs.detach().cpu().tolist()

            for i in range(len(pred_cpu)):
                if len(samples) >= MAX_SAVED_EXAMPLES:
                    break

                samples.append(
                    {
                        "true_action": true_cpu[i],
                        "pred_action": pred_cpu[i],
                        "active_probs": [round(float(v), 4) for v in probs_cpu[i]],
                    }
                )

    averaged = {
        key: scalar_sums[key] / max(1, batches)
        for key in scalar_sums
    }

    tp = merged["tp"]
    fp = merged["fp"]
    fn = merged["fn"]
    tn = merged["tn"]

    global_active_precision = tp / max(1, tp + fp)
    global_active_recall = tp / max(1, tp + fn)
    global_active_f1 = (2 * global_active_precision * global_active_recall) / max(
        1e-8,
        global_active_precision + global_active_recall,
    )

    print()
    print("Pipeline v1 Eval")
    print("----------------")
    print(f"joint_acc:                {averaged['joint_acc']:.4f}")
    print(f"exact_acc:                {averaged['exact_acc']:.4f}")
    print(f"active_joint_acc:         {averaged['active_joint_acc']:.4f}")
    print(f"active_precision:         {global_active_precision:.4f}")
    print(f"active_recall:            {global_active_recall:.4f}")
    print(f"active_f1:                {global_active_f1:.4f}")
    print(f"pred_active_avg:          {averaged['pred_active_avg']:.2f}")
    print(f"true_active_avg:          {averaged['true_active_avg']:.2f}")
    print(f"value_acc_on_true_active: {averaged['value_acc_on_true_active']:.4f}")

    result = {
        "active_threshold": ACTIVE_THRESHOLD,
        "averaged": averaged,
        "global_active": {
            "precision": global_active_precision,
            "recall": global_active_recall,
            "f1": global_active_f1,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        },
        "samples": samples,
    }

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print()
    print(f"Saved sample predictions to: {OUT_PATH}")


if __name__ == "__main__":
    main()
