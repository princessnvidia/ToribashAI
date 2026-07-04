#!/usr/bin/env python3
from pathlib import Path
import json
from collections import Counter

import torch
import torch.nn as nn

BASE = Path.home() / "Documents/ToribashAI"

REPLAY_JSON = BASE / "datasets" / "parkour_json" / "00e9325d61e9_[p] karbn - mirrors edge pk v9.json"
MODEL_PATH = BASE / "models" / "parkour_gru_v4_mod_split.pt"
OUT = BASE / "models" / "predicted_actions_sample.json"

SEQ_LEN = 8
STATE_DIM = 273
JOINTS = 20
CLASSES = 5


class GRUPolicy(nn.Module):
    def __init__(self, hidden_size=128, num_layers=1, dropout=0.25):
        super().__init__()

        self.gru = nn.GRU(
            input_size=STATE_DIM,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.0,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, JOINTS * CLASSES),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits.view(-1, JOINTS, CLASSES)


def flatten(values):
    out = []
    for item in values:
        if isinstance(item, list):
            out.extend(flatten(item))
        else:
            out.append(item)
    return out


def state_vector(player):
    return (
        flatten(player.get("pos", [])) +
        flatten(player.get("qat", [])) +
        flatten(player.get("linvel", [])) +
        flatten(player.get("angvel", []))
    )


def joint_vector(player):
    joints = player.get("joints", {})
    return [joints.get(str(i), 0) for i in range(JOINTS)]


def load_replay_sequences(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", {})
    frame_ids = sorted(int(k) for k in frames.keys())

    rows = []

    for fid in frame_ids:
        player = frames[str(fid)].get("players", {}).get("0")
        if not player:
            continue

        state = state_vector(player)
        action = joint_vector(player)

        if len(state) != STATE_DIM or len(action) != JOINTS:
            continue

        rows.append({
            "frame": fid,
            "state": state,
            "true_action": action,
        })

    sequences = []

    for i in range(len(rows) - SEQ_LEN + 1):
        window = rows[i:i + SEQ_LEN]

        sequences.append({
            "start_frame": window[0]["frame"],
            "end_frame": window[-1]["frame"],
            "states": [r["state"] for r in window],
            "true_action": window[-1]["true_action"],
        })

    return data, sequences


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)
    print("Replay:", REPLAY_JSON)
    print("Model:", MODEL_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=device)

    model = GRUPolicy(
        hidden_size=checkpoint.get("hidden_size", 128),
        num_layers=checkpoint.get("num_layers", 1),
        dropout=checkpoint.get("dropout", 0.25),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean = checkpoint["mean"].to(device)
    std = checkpoint["std"].to(device)

    replay_data, sequences = load_replay_sequences(REPLAY_JSON)

    print("Sequences:", len(sequences))

    results = []
    correct_joints = 0
    total_joints = 0
    exact_actions = 0

    pred_counter = Counter()
    true_counter = Counter()

    with torch.no_grad():
        for seq in sequences:
            x = torch.tensor(
                [seq["states"]],
                dtype=torch.float32,
                device=device
            )

            x = (x - mean.view(1, 1, -1)) / std.view(1, 1, -1)

            logits = model(x)
            pred = logits.argmax(dim=-1)[0].cpu().tolist()
            true = seq["true_action"]

            matches = [int(p == t) for p, t in zip(pred, true)]

            correct_joints += sum(matches)
            total_joints += JOINTS

            if pred == true:
                exact_actions += 1

            for p in pred:
                pred_counter[p] += 1

            for t in true:
                true_counter[t] += 1

            results.append({
                "start_frame": seq["start_frame"],
                "end_frame": seq["end_frame"],
                "true_action": true,
                "pred_action": pred,
                "joint_matches": matches,
                "joint_accuracy": sum(matches) / JOINTS,
                "exact": pred == true,
            })

    summary = {
        "replay_json": str(REPLAY_JSON),
        "model": str(MODEL_PATH),
        "fightname": replay_data.get("metadata", {}).get("fightname", ""),
        "mod": replay_data.get("metadata", {}).get("mod", ""),
        "sequences": len(sequences),
        "joint_accuracy": correct_joints / total_joints if total_joints else 0,
        "exact_action_accuracy": exact_actions / len(sequences) if sequences else 0,
        "true_values": true_counter.most_common(),
        "predicted_values": pred_counter.most_common(),
        "first_predictions": results[:50],
    }

    OUT.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Sortie:", OUT)
    print("Joint accuracy:", summary["joint_accuracy"])
    print("Exact action accuracy:", summary["exact_action_accuracy"])
    print("Vraies valeurs:", true_counter.most_common())
    print("Valeurs prédites:", pred_counter.most_common())

    print("\nPremières prédictions:")
    for row in results[:10]:
        print("Frame", row["end_frame"])
        print("  vrai:", row["true_action"])
        print("  pred:", row["pred_action"])
        print("  acc :", row["joint_accuracy"])


if __name__ == "__main__":
    main()
