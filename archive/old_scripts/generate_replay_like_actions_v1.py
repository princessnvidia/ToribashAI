#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

import torch
from torch import nn


PROJECT_DIR = Path.home() / "Documents" / "ToribashAI"

DATASET_PATH = PROJECT_DIR / "datasets" / "ml" / "parkour_active_joints_len8.jsonl"

ACTIVE_MODEL_PATH = PROJECT_DIR / "models" / "parkour_active_joints_gru_v4_weight070.pt"
VALUE_MODEL_PATH = PROJECT_DIR / "models" / "parkour_active_values_gru_v1.pt"

OUTPUT_PATH = PROJECT_DIR / "models" / "generated_replay_like_actions_v1.json"

SEQ_LEN = 8
STATE_SIZE = 273
NUM_JOINTS = 20

MAX_ACTIVE_JOINTS = 3
MAX_FRAMES = 500


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
            input_size=STATE_SIZE + NUM_JOINTS,
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


def make_value_input(states_seq, joint_id):
    states = torch.tensor(states_seq, dtype=torch.float32).unsqueeze(0)

    joint_onehot = torch.zeros(1, NUM_JOINTS)
    joint_onehot[0, joint_id] = 1.0

    joint_seq = joint_onehot.unsqueeze(1).repeat(1, SEQ_LEN, 1)

    return torch.cat([states, joint_seq], dim=2)


def predict_action(states_seq, active_model, value_model):
    states_tensor = torch.tensor(states_seq, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        active_logits = active_model(states_tensor)
        active_probs = torch.sigmoid(active_logits)[0]

    predicted_action = [0] * NUM_JOINTS

    top_joint_ids = torch.topk(
        active_probs,
        k=MAX_ACTIVE_JOINTS,
    ).indices.tolist()

    for joint_id in top_joint_ids:
        value_input = make_value_input(states_seq, joint_id)

        with torch.no_grad():
            value_logits = value_model(value_input)

        value_class = value_logits.argmax(dim=1).item()
        predicted_action[joint_id] = value_class + 1

    return predicted_action, [round(float(v), 4) for v in active_probs.tolist()]


def joint_accuracy(true_action, predicted_action):
    correct = 0

    for true_value, pred_value in zip(true_action, predicted_action):
        if int(true_value) == int(pred_value):
            correct += 1

    return correct / NUM_JOINTS


def main():
    print(f"Dataset: {DATASET_PATH}")
    print(f"Max active joints: {MAX_ACTIVE_JOINTS}")
    print(f"Max frames: {MAX_FRAMES}")

    active_ckpt = torch.load(ACTIVE_MODEL_PATH, map_location="cpu")
    active_cfg = active_ckpt["config"]

    active_model = ActiveJointsGRU(
        input_size=active_cfg["input_size"],
        hidden_size=active_cfg["hidden_size"],
        output_size=active_cfg["output_size"],
        num_layers=active_cfg["num_layers"],
        dropout=active_cfg["dropout"],
    )

    active_model.load_state_dict(active_ckpt["model_state_dict"])
    active_model.eval()

    value_ckpt = torch.load(VALUE_MODEL_PATH, map_location="cpu")

    value_model = ActiveValueGRU()
    value_model.load_state_dict(value_ckpt["model_state_dict"])
    value_model.eval()

    frames = []

    true_value_counter = Counter()
    predicted_value_counter = Counter()

    true_active_counts = []
    predicted_active_counts = []
    joint_accuracies = []

    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if len(frames) >= MAX_FRAMES:
                break

            if not line.strip():
                continue

            obj = json.loads(line)

            states = obj.get("states") or obj.get("state_seq")
            true_action = obj.get("action")

            if states is None or true_action is None:
                continue

            true_action = [int(v) for v in true_action]

            predicted_action, active_probs = predict_action(
                states,
                active_model,
                value_model,
            )

            true_active_count = sum(v != 0 for v in true_action)
            predicted_active_count = sum(v != 0 for v in predicted_action)

            acc = joint_accuracy(true_action, predicted_action)

            true_active_counts.append(true_active_count)
            predicted_active_counts.append(predicted_active_count)
            joint_accuracies.append(acc)

            for v in true_action:
                true_value_counter[int(v)] += 1

            for v in predicted_action:
                predicted_value_counter[int(v)] += 1

            frames.append(
                {
                    "frame_index": len(frames),
                    "true_action": true_action,
                    "predicted_action": predicted_action,
                    "true_active_count": true_active_count,
                    "predicted_active_count": predicted_active_count,
                    "joint_accuracy": round(acc, 4),
                    "active_probs": active_probs,
                }
            )

    summary = {
        "frame_count": len(frames),
        "max_active_joints": MAX_ACTIVE_JOINTS,
        "avg_true_active_count": sum(true_active_counts) / max(1, len(true_active_counts)),
        "avg_predicted_active_count": sum(predicted_active_counts) / max(1, len(predicted_active_counts)),
        "avg_joint_accuracy": sum(joint_accuracies) / max(1, len(joint_accuracies)),
        "true_value_distribution": sorted(true_value_counter.items()),
        "predicted_value_distribution": sorted(predicted_value_counter.items()),
    }

    output = {
        "metadata": {
            "description": "Replay-like predicted Toribash actions generated from ML sequence dataset.",
            "dataset": str(DATASET_PATH),
            "active_model": str(ACTIVE_MODEL_PATH),
            "value_model": str(VALUE_MODEL_PATH),
            "seq_len": SEQ_LEN,
            "state_size": STATE_SIZE,
            "num_joints": NUM_JOINTS,
            "max_active_joints": MAX_ACTIVE_JOINTS,
            "max_frames": MAX_FRAMES,
        },
        "summary": summary,
        "frames": frames,
    }

    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print()
    print(f"Saved: {OUTPUT_PATH}")

    print()
    print("Summary:")
    print(json.dumps(summary, indent=2))

    print()
    print("First frame:")
    print(json.dumps(frames[0], indent=2))


if __name__ == "__main__":
    main()
