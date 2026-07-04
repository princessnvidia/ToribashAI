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

OUT_PATH = PROJECT_DIR / "models" / "pipeline_v1_detailed_eval.json"

SEQ_LEN = 8
STATE_SIZE = 273
NUM_JOINTS = 20
VALUE_INPUT_SIZE = STATE_SIZE + NUM_JOINTS
ACTIVE_THRESHOLD = 0.45
BATCH_SIZE = 128
SEED = 42


class SequenceDataset(Dataset):
    def __init__(self, path):
        self.items = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue

                obj = json.loads(line)
                states = obj.get("states") or obj.get("state_seq")
                action = obj.get("action")

                if states is None or action is None:
                    continue

                action = [int(v) for v in action]

                if len(states) != SEQ_LEN or len(action) != NUM_JOINTS:
                    continue

                self.items.append((states, action))

        if not self.items:
            raise RuntimeError(f"Dataset vide ou invalide: {path}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        states, action = self.items[idx]
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(action, dtype=torch.long),
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
        return self.head(out[:, -1, :])


def split_val_dataset(path):
    random.seed(SEED)

    full = SequenceDataset(path)
    indices = list(range(len(full)))
    random.shuffle(indices)

    val_size = max(1, int(len(indices) * 0.15))
    val_indices = set(indices[:val_size])

    val_ds = SequenceDataset.__new__(SequenceDataset)
    val_ds.items = [
        item for i, item in enumerate(full.items)
        if i in val_indices
    ]

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

    return torch.cat([states_batch, joint_seq], dim=2)


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

        pred_values = value_logits.argmax(dim=1) + 1
        pred_actions[row_mask, joint_id] = pred_values

    return pred_actions, active_probs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
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

    confusion = torch.zeros(5, 5, dtype=torch.long)

    total_positions = 0
    exact_rows = 0
    total_rows = 0

    active_tp = 0
    active_fp = 0
    active_fn = 0
    active_tn = 0

    value_eval_total = 0
    value_eval_correct = 0

    true_active_value_total = 0
    true_active_full_correct = 0
    true_active_missed = 0
    true_active_wrong_value = 0

    pred_counter = Counter()
    true_counter = Counter()

    joint_total = Counter()
    joint_full_correct = Counter()
    joint_active_detected = Counter()
    joint_value_correct_when_detected = Counter()

    with torch.no_grad():
        for states, true_actions in loader:
            true_actions = true_actions.to(device)

            pred_actions, _active_probs = predict_pipeline(
                states,
                active_model,
                value_model,
                device,
            )

            total_rows += true_actions.shape[0]
            total_positions += true_actions.numel()

            exact_rows += (
                (pred_actions == true_actions).sum(dim=1) == NUM_JOINTS
            ).sum().item()

            pred_active = pred_actions != 0
            true_active = true_actions != 0

            active_tp += ((pred_active == 1) & (true_active == 1)).sum().item()
            active_fp += ((pred_active == 1) & (true_active == 0)).sum().item()
            active_fn += ((pred_active == 0) & (true_active == 1)).sum().item()
            active_tn += ((pred_active == 0) & (true_active == 0)).sum().item()

            both_active = pred_active & true_active

            value_eval_total += both_active.sum().item()
            value_eval_correct += (
                pred_actions[both_active] == true_actions[both_active]
            ).sum().item()

            true_active_value_total += true_active.sum().item()
            true_active_full_correct += (
                pred_actions[true_active] == true_actions[true_active]
            ).sum().item()

            true_active_missed += (
                (pred_actions == 0) & (true_actions != 0)
            ).sum().item()

            true_active_wrong_value += (
                (pred_actions != 0)
                & (true_actions != 0)
                & (pred_actions != true_actions)
            ).sum().item()

            for t, p in zip(
                true_actions.detach().cpu().flatten().tolist(),
                pred_actions.detach().cpu().flatten().tolist(),
            ):
                true_counter[int(t)] += 1
                pred_counter[int(p)] += 1
                confusion[int(t), int(p)] += 1

            for batch_idx in range(true_actions.shape[0]):
                for joint_id in range(NUM_JOINTS):
                    t = int(true_actions[batch_idx, joint_id].item())
                    p = int(pred_actions[batch_idx, joint_id].item())

                    joint_total[joint_id] += 1

                    if t == p:
                        joint_full_correct[joint_id] += 1

                    if t != 0 and p != 0:
                        joint_active_detected[joint_id] += 1

                        if t == p:
                            joint_value_correct_when_detected[joint_id] += 1

    joint_acc = confusion.diag().sum().item() / max(1, total_positions)
    exact_acc = exact_rows / max(1, total_rows)

    active_precision = active_tp / max(1, active_tp + active_fp)
    active_recall = active_tp / max(1, active_tp + active_fn)
    active_f1 = (2 * active_precision * active_recall) / max(
        1e-8,
        active_precision + active_recall,
    )

    value_acc_when_detected = value_eval_correct / max(1, value_eval_total)
    value_acc_on_true_active = true_active_full_correct / max(1, true_active_value_total)

    print()
    print("Pipeline v1 Detailed Eval")
    print("-------------------------")
    print(f"joint_acc:                    {joint_acc:.4f}")
    print(f"exact_acc:                    {exact_acc:.4f}")
    print(f"active_precision:             {active_precision:.4f}")
    print(f"active_recall:                {active_recall:.4f}")
    print(f"active_f1:                    {active_f1:.4f}")
    print(f"value_acc_when_detected:      {value_acc_when_detected:.4f}")
    print(f"value_acc_on_true_active:     {value_acc_on_true_active:.4f}")
    print(f"true_active_missed:           {true_active_missed}")
    print(f"true_active_wrong_value:      {true_active_wrong_value}")
    print(f"true_active_full_correct:     {true_active_full_correct}")
    print(f"true_active_total:            {true_active_value_total}")

    print()
    print("True action distribution:")
    print(sorted(true_counter.items()))

    print()
    print("Predicted action distribution:")
    print(sorted(pred_counter.items()))

    print()
    print("Confusion matrix rows=true, cols=pred")
    print("       pred0  pred1  pred2  pred3  pred4")
    for i in range(5):
        row = confusion[i].tolist()
        print(
            f"true{i} "
            f"{row[0]:7d} {row[1]:7d} {row[2]:7d} {row[3]:7d} {row[4]:7d}"
        )

    print()
    print("Joint diagnostics:")
    per_joint = []
    for joint_id in range(NUM_JOINTS):
        full_acc = joint_full_correct[joint_id] / max(1, joint_total[joint_id])

        detected = joint_active_detected[joint_id]
        value_ok = joint_value_correct_when_detected[joint_id]
        value_acc = value_ok / max(1, detected)

        per_joint.append(
            {
                "joint": joint_id,
                "full_acc": full_acc,
                "value_acc_when_detected": value_acc,
                "detected_active_pairs": int(detected),
                "value_correct_when_detected": int(value_ok),
            }
        )

        print(
            f"joint {joint_id:02d}: "
            f"full_acc={full_acc:.4f} | "
            f"value_acc_when_detected={value_acc:.4f} | "
            f"detected_active_pairs={detected}"
        )

    summary = {
        "active_threshold": ACTIVE_THRESHOLD,
        "joint_acc": joint_acc,
        "exact_acc": exact_acc,
        "active_precision": active_precision,
        "active_recall": active_recall,
        "active_f1": active_f1,
        "value_acc_when_detected": value_acc_when_detected,
        "value_acc_on_true_active": value_acc_on_true_active,
        "true_active_missed": int(true_active_missed),
        "true_active_wrong_value": int(true_active_wrong_value),
        "true_active_full_correct": int(true_active_full_correct),
        "true_active_total": int(true_active_value_total),
        "true_distribution": sorted(true_counter.items()),
        "pred_distribution": sorted(pred_counter.items()),
        "confusion": confusion.tolist(),
        "per_joint": per_joint,
    }

    with OUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Saved eval to: {OUT_PATH}")


if __name__ == "__main__":
    main()
