#!/usr/bin/env python3
from pathlib import Path
import shutil
import torch
import torch.nn as nn

PROJECT = Path.home() / "Documents" / "ToribashAI"

MODEL_PATH = PROJECT / "models" / "run_legs_gru_v1.pt"
OUT_DIR = PROJECT / "models" / "run_legs_generated_v1"

TORIBASH_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
)
TORIBASH_REPLAY_DIR = TORIBASH_DIR / "replay"

NUM_REPLAYS = 20
STEPS = 45

MATCHFRAMES = 1000
TURNFRAMES = 20
MOD_NAME = "ToribashAI/toribashai_goal_flat_v1.tbm"

START_Z = 5.40
TARGET_Y = -12.0

LEG_JOINTS = [14, 15, 16, 17, 18, 19]
NUM_LEG_JOINTS = 6
NUM_CLASSES = 5
INPUT_DIM = 12
HIDDEN_SIZE = 96
NUM_LAYERS = 1
DROPOUT = 0.15


class RunLegsGRU(nn.Module):
    def __init__(self):
        super().__init__()

        self.gru = nn.GRU(
            input_size=INPUT_DIM,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 96),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(96, NUM_LEG_JOINTS * NUM_CLASSES),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits.view(-1, NUM_LEG_JOINTS, NUM_CLASSES)


def hold_all(player_id):
    return (
        f"JOINT {player_id}; "
        "0 3 1 3 2 3 3 3 4 3 5 3 6 3 7 3 8 3 9 3 "
        "10 3 11 3 12 3 13 3 14 3 15 3 16 3 17 3 18 3 19 3"
    )


def full_action_from_legs(legs):
    actions = [3] * 20

    actions[0] = 3
    actions[1] = 3
    actions[2] = 3
    actions[3] = 1
    actions[12] = 3
    actions[13] = 3

    for jid, val in zip(LEG_JOINTS, legs):
        actions[jid] = int(val)

    return actions


def action_line(player_id, actions):
    parts = []
    for jid, val in enumerate(actions):
        parts.append(str(jid))
        parts.append(str(int(val)))
    return f"JOINT {player_id}; " + " ".join(parts)


def sample_from_logits(logits, temperature=0.85):
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def make_feature_vector(leg_now, speed, leg_activity, support_change, lean, z_min, z_range):
    return (
        [float(v) for v in leg_now]
        + [
            float(speed),
            float(leg_activity),
            float(support_change),
            float(lean),
            float(z_min),
            float(z_range),
        ]
    )


def generate_one(index, model, ckpt):
    torch.manual_seed(20000 + index)

    mean = ckpt["mean"]
    std = ckpt["std"]

    name = f"ToribashAI_run_legs_gru_{index:03d}"
    path = OUT_DIR / f"{name}.rpl"

    speed = 10.0 + (index % 5) * 0.75
    leg_activity = 0.30
    support_change = 0.60
    lean = 0.22
    z_min = 5.6
    z_range = 1.4

    leg_now = [3, 3, 3, 3, 3, 3]
    history = []

    for _ in range(8):
        history.append(
            make_feature_vector(
                leg_now,
                speed,
                leg_activity,
                support_change,
                lean,
                z_min,
                z_range,
            )
        )

    generated_legs = []

    for _ in range(STEPS):
        x = torch.tensor([history[-8:]], dtype=torch.float32)
        x = (x - mean) / std

        with torch.no_grad():
            logits = model(x)[0]

        legs = sample_from_logits(logits, temperature=0.85).tolist()

        generated_legs.append(legs)
        leg_now = legs

        history.append(
            make_feature_vector(
                leg_now,
                speed,
                leg_activity,
                support_change,
                lean,
                z_min,
                z_range,
            )
        )

    lines = [
        "#SCORE 0 0",
        "#WIN 2",
        "VERSION 12",
        f"FIGHTNAME 0; {name}",
        "BOUT 0; ToribashAI",
        "BOUT 1; Target",
        "AUTHOR 0; ToribashAI",
        "AUTHOR 1; ToribashAI",
        f"ENGAGE 0; 0.000000 0.000000 {START_Z:.6f} 0 0 0",
        f"ENGAGE 1; 0.000000 {TARGET_Y:.6f} {START_Z:.6f} 0 0 0",
        (
            f"NEWGAME 1;{MATCHFRAMES} {TURNFRAMES} 200000 0 0 8 200 0 1 "
            f"{MOD_NAME} 0 0 250 500 1000 0 1 0 2 0 0 0 0 0 0 "
            f"0.000000 0.000000 -30.000000 0 0 0 0 8"
        ),
    ]

    frame = 0
    lines.append(f"FRAME {frame}; 0 0 0 0")
    lines.append(hold_all(0))
    lines.append(hold_all(1))

    for _ in range(2):
        frame += TURNFRAMES
        lines.append(f"FRAME {frame}; 0 0 0 0")
        lines.append(hold_all(0))
        lines.append(hold_all(1))

    for legs in generated_legs:
        frame += TURNFRAMES
        actions = full_action_from_legs(legs)
        lines.append(f"FRAME {frame}; 0 0 0 0")
        lines.append(action_line(0, actions))
        lines.append(hold_all(1))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def clean_generated_dirs():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TORIBASH_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    for p in OUT_DIR.glob("ToribashAI_run_legs_gru_*.rpl"):
        p.unlink()

    for p in TORIBASH_REPLAY_DIR.glob("ToribashAI_run_legs_gru_*.rpl"):
        p.unlink()


def copy_to_toribash(paths):
    copied = []

    for src in paths:
        dst = TORIBASH_REPLAY_DIR / src.name
        shutil.copy2(src, dst)
        copied.append(dst)

    return copied


def main():
    clean_generated_dirs()

    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    model = RunLegsGRU()
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print("Model chargé:", MODEL_PATH)
    print("Output:", OUT_DIR)
    print("Toribash replay:", TORIBASH_REPLAY_DIR)
    print()

    generated_paths = []

    for i in range(NUM_REPLAYS):
        path = generate_one(i, model, ckpt)
        generated_paths.append(path)
        print("Generated:", path.name)

    copied_paths = copy_to_toribash(generated_paths)

    print()
    print("Copiés dans Toribash:")
    for path in copied_paths:
        print(path.name)

    print()
    print("Terminé.")
    print(f"Dossier local: {OUT_DIR}")
    print(f"Dossier Toribash: {TORIBASH_REPLAY_DIR}")


if __name__ == "__main__":
    main()
