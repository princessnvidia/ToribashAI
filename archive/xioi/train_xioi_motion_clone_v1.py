#!/usr/bin/env python3
import json
import math
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path.home() / "Documents" / "ToribashAI"

SOURCE = ROOT / "parsed/#Xioi#Pk - YpSkA.json"
MODEL_PATH = ROOT / "models/xioi_motion_clone_v1.pt"
SUMMARY_PATH = ROOT / "models/xioi_motion_clone_v1_summary.json"

PLAYER_ID = "0"
MAX_FRAME = 427
NUM_CLASSES = 5
ACTION_DIM = 20

EPOCHS = 2000
LR = 1e-3
HIDDEN = 256


def fix_target(v):
    v = int(v)
    if v < 0:
        return 0
    if v > 4:
        return 4
    return v


def load_dataset():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    frames = data["frames"]

    xs = []
    ys = []
    frame_ids = []

    usable = sorted(int(k) for k in frames.keys() if int(k) <= MAX_FRAME)

    for fid in usable:
        player = frames[str(fid)].get("players", {}).get(PLAYER_ID)
        if not player:
            continue

        joints = player.get("joints", {})
        action = [fix_target(joints.get(str(j), 0)) for j in range(ACTION_DIM)]

        t = fid / MAX_FRAME
        x = [
            t,
            math.sin(t * math.pi * 2),
            math.cos(t * math.pi * 2),
            math.sin(t * math.pi * 4),
            math.cos(t * math.pi * 4),
            math.sin(t * math.pi * 8),
            math.cos(t * math.pi * 8),
        ]

        xs.append(x)
        ys.append(action)
        frame_ids.append(fid)

    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long), frame_ids


class MotionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Linear(HIDDEN, ACTION_DIM * NUM_CLASSES),
        )

    def forward(self, x):
        return self.net(x).view(-1, ACTION_DIM, NUM_CLASSES)


def main():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    x, y, frame_ids = load_dataset()

    print("Frames:", len(frame_ids))
    print("First frame:", frame_ids[0])
    print("Last frame:", frame_ids[-1])
    print("X:", x.shape)
    print("Y:", y.shape)

    model = MotionMLP()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = 999999
    best_acc = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad()

        logits = model(x)
        loss = loss_fn(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))

        loss.backward()
        opt.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            joint_acc = (pred == y).float().mean().item()
            exact_acc = (pred == y).all(dim=1).float().mean().item()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_acc = joint_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "max_frame": MAX_FRAME,
                    "action_dim": ACTION_DIM,
                    "num_classes": NUM_CLASSES,
                    "hidden": HIDDEN,
                    "frame_ids": frame_ids,
                    "source": str(SOURCE),
                },
                MODEL_PATH,
            )

        if epoch % 100 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:04d} | "
                f"loss={loss.item():.6f} | "
                f"joint_acc={joint_acc:.4f} | "
                f"exact={exact_acc:.4f}"
            )

        if exact_acc >= 0.995:
            print("Presque parfait, stop.")
            break

    summary = {
        "source": str(SOURCE),
        "model": str(MODEL_PATH),
        "frames": len(frame_ids),
        "first_frame": frame_ids[0],
        "last_frame": frame_ids[-1],
        "best_loss": best_loss,
        "best_joint_acc": best_acc,
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("BEST:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
