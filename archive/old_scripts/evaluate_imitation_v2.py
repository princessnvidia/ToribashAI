#!/usr/bin/env python3
from pathlib import Path
import json
from collections import Counter

import torch
import torch.nn as nn

BASE = Path.home() / "Documents/ToribashAI"

DATASET = BASE / "datasets" / "ml" / "parkour_transitions_clean.jsonl"
MODEL_PATH = BASE / "models" / "parkour_mlp_v4_soft_weighted.pt"
OUT = BASE / "models" / "parkour_mlp_v4_soft_weighted_eval.json"

STATE_DIM = 273
JOINTS = 20
CLASSES = 5


class MLPPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 512),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(512, JOINTS * CLASSES),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits.view(-1, JOINTS, CLASSES)


def load_rows(path):
    states = []
    actions = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            states.append(row["state"])
            actions.append(row["action"])

    return (
        torch.tensor(states, dtype=torch.float32),
        torch.tensor(actions, dtype=torch.long),
    )


def counter_to_ratios(counter):
    total = sum(counter.values())

    return [
        {
            "value": value,
            "count": count,
            "ratio": count / total if total else 0.0,
        }
        for value, count in counter.most_common()
    ]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Model:", MODEL_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = MLPPolicy().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint["mean"].to(device)
    std = checkpoint["std"].to(device)

    states, actions = load_rows(DATASET)

    states = states.to(device)
    actions = actions.to(device)

    states = (states - mean) / std

    print("Transitions:", len(states))

    with torch.no_grad():
        logits = model(states)
        pred = logits.argmax(dim=-1)

    correct = pred == actions

    joint_accs = []
    joint_confusions = {}
    true_by_joint = {}
    pred_by_joint = {}

    for joint_id in range(JOINTS):
        c = correct[:, joint_id].float().mean().item()
        joint_accs.append(c)

        confusion = Counter()
        true_counter = Counter()
        pred_counter = Counter()

        real = actions[:, joint_id].cpu().tolist()
        guessed = pred[:, joint_id].cpu().tolist()

        for r, g in zip(real, guessed):
            confusion[f"{r}->{g}"] += 1
            true_counter[r] += 1
            pred_counter[g] += 1

        joint_confusions[str(joint_id)] = confusion.most_common(20)
        true_by_joint[str(joint_id)] = counter_to_ratios(true_counter)
        pred_by_joint[str(joint_id)] = counter_to_ratios(pred_counter)

    exact_acc = correct.all(dim=1).float().mean().item()
    joint_acc = correct.float().mean().item()

    true_values = Counter(actions.cpu().reshape(-1).tolist())
    pred_values = Counter(pred.cpu().reshape(-1).tolist())

    summary = {
        "dataset": str(DATASET),
        "model": str(MODEL_PATH),
        "device": device,
        "transitions": len(states),
        "joint_accuracy_global": joint_acc,
        "exact_action_accuracy": exact_acc,
        "joint_accuracy_by_joint": {
            str(i): joint_accs[i]
            for i in range(JOINTS)
        },
        "true_action_values": counter_to_ratios(true_values),
        "predicted_action_values": counter_to_ratios(pred_values),
        "true_values_by_joint": true_by_joint,
        "predicted_values_by_joint": pred_by_joint,
        "confusion_by_joint_top": joint_confusions,
        "checkpoint_extra": {
            "state_dim": checkpoint.get("state_dim"),
            "joints": checkpoint.get("joints"),
            "classes": checkpoint.get("classes"),
            "class_counts": checkpoint.get("class_counts"),
            "weight_power": checkpoint.get("weight_power"),
        },
    }

    OUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Global joint acc:", joint_acc)
    print("Exact action acc:", exact_acc)
    print("Éval sauvée:", OUT)

    print("\nAccuracy par joint:")
    for i, acc in enumerate(joint_accs):
        print(f"joint {i:02d}: {acc:.4f}")

    print("\nVraies valeurs:", true_values.most_common())
    print("Valeurs prédites:", pred_values.most_common())


if __name__ == "__main__":
    main()
