#!/usr/bin/env python3
import json
from pathlib import Path

import torch
from torch import nn


PROJECT_DIR = Path.home() / "Documents" / "ToribashAI"

DATASET_PATH = PROJECT_DIR / "datasets" / "ml" / "parkour_active_joints_len8.jsonl"

ACTIVE_MODEL_PATH = PROJECT_DIR / "models" / "parkour_active_joints_gru_v4_weight070.pt"
VALUE_MODEL_PATH = PROJECT_DIR / "models" / "parkour_active_values_gru_v1.pt"

OUTPUT_PATH = PROJECT_DIR / "models" / "generated_replay_v1.json"

SEQ_LEN = 8
NUM_JOINTS = 20
STATE_SIZE = 273
ACTIVE_THRESHOLD = 0.45


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


def build_state_vector(frame):
    vec = []

    players_obj = frame["players"]

    if isinstance(players_obj, dict):
        players = [
            players_obj[k]
            for k in sorted(players_obj.keys(), key=lambda x: int(x))
        ]
    else:
        players = players_obj

    for player in players:
        vec.extend(player.get("pos", []))
        vec.extend(player.get("linvel", []))
        vec.extend(player.get("angvel", []))

        joints_obj = player.get("joints", [])

        if isinstance(joints_obj, dict):
            joints = [
                joints_obj[k]
                for k in sorted(joints_obj.keys(), key=lambda x: int(x))
            ]
        else:
            joints = joints_obj

        for joint in joints:
            if isinstance(joint, dict):
                vec.append(joint.get("state", 0))
            else:
                vec.append(int(joint))

    if len(vec) != STATE_SIZE:
        raise ValueError(
            f"State vector has wrong size: {len(vec)} instead of {STATE_SIZE}. "
            f"Frame keys: {frame.keys()}"
        )

    return vec


def make_value_input(states_seq, joint_id):
    states = torch.tensor(
        states_seq,
        dtype=torch.float32,
    ).unsqueeze(0)

    joint_onehot = torch.zeros(1, NUM_JOINTS)
    joint_onehot[0, joint_id] = 1.0

    joint_seq = joint_onehot.unsqueeze(1).repeat(
        1,
        SEQ_LEN,
        1,
    )

    return torch.cat(
        [states, joint_seq],
        dim=2,
    )


def main():
    print(f"Replay: {REPLAY_PATH}")

    active_ckpt = torch.load(
        ACTIVE_MODEL_PATH,
        map_location="cpu",
    )

    active_cfg = active_ckpt["config"]

    active_model = ActiveJointsGRU(
        input_size=active_cfg["input_size"],
        hidden_size=active_cfg["hidden_size"],
        output_size=active_cfg["output_size"],
        num_layers=active_cfg["num_layers"],
        dropout=active_cfg["dropout"],
    )

    active_model.load_state_dict(
        active_ckpt["model_state_dict"]
    )

    active_model.eval()

    value_ckpt = torch.load(
        VALUE_MODEL_PATH,
        map_location="cpu",
    )

    value_model = ActiveValueGRU()

    value_model.load_state_dict(
        value_ckpt["model_state_dict"]
    )

    value_model.eval()

    replay = json.loads(
        REPLAY_PATH.read_text(
            encoding="utf-8"
        )
    )

    frames_obj = replay["frames"]

    if isinstance(frames_obj, dict):
        frames = [
            frames_obj[k]
            for k in sorted(frames_obj.keys(), key=lambda x: int(x))
        ]
    else:
        frames = frames_obj

    state_vectors = []

    for frame in frames:
        state_vectors.append(
            build_state_vector(frame)
        )

    outputs = []

    for i in range(
        SEQ_LEN,
        len(state_vectors)
    ):
        states_seq = state_vectors[
            i - SEQ_LEN:i
        ]

        states_tensor = torch.tensor(
            states_seq,
            dtype=torch.float32,
        ).unsqueeze(0)

        with torch.no_grad():
            active_logits = active_model(
                states_tensor
            )

            active_probs = torch.sigmoid(
                active_logits
            )[0]

        predicted_action = [0] * NUM_JOINTS

        for joint_id in range(NUM_JOINTS):

            if (
                active_probs[joint_id]
                < ACTIVE_THRESHOLD
            ):
                continue

            value_input = make_value_input(
                states_seq,
                joint_id,
            )

            with torch.no_grad():
                value_logits = value_model(
                    value_input
                )

            value_class = (
                value_logits.argmax(
                    dim=1
                ).item()
            )

            predicted_action[joint_id] = (
                value_class + 1
            )

        outputs.append(
            {
                "frame_index": i,
                "predicted_action": predicted_action,
                "active_count": sum(
                    v != 0
                    for v in predicted_action
                )
            }
        )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            outputs,
            f,
            indent=2,
        )

    print()
    print(
        f"Generated actions: {len(outputs)}"
    )

    print(
        f"Saved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
