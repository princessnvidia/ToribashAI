#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path.home() / "Documents/ToribashAI"
GEN = ROOT / "generated_replays"
DATASET = ROOT / "datasets/ml/xioi_loop_len265_v52_sequences.jsonl"
MODEL = ROOT / "models/xioi_loop_len265_gru_v52.pt"
TEMPLATE = GEN / "xioi_loop_len265_champion_v51.rpl"
OUT = GEN / "xioi_loop_len265_gru_generated_v52.rpl"
ACTIONS_JSON = GEN / "xioi_loop_len265_gru_generated_v52_actions.json"
STEAM = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
REPLAY_ROOT = STEAM / "replay"
REPLAY_PARKOUR = REPLAY_ROOT / "parkour"

FRAME_RE = re.compile(r"^FRAME\s+(\d+);")
JOINT0_RE = re.compile(r"^JOINT\s+0;")

STATE_DIM = 20
ACTION_DIM = 20
CLASSES = 5


class GRUAction(nn.Module):
    def __init__(self, state_dim, hidden, layers):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden, num_layers=layers, batch_first=True, dropout=0.10 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, ACTION_DIM * CLASSES))
    def forward(self, x):
        z, _ = self.gru(x)
        return self.head(z[:, -1]).view(-1, ACTION_DIM, CLASSES)


def load_rows():
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_model(device):
    ckpt = torch.load(MODEL, map_location=device)
    model = GRUAction(ckpt.get("state_dim", 20), ckpt.get("hidden", 192), ckpt.get("layers", 2)).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    print("Loaded:", MODEL, "epoch=", ckpt.get("epoch"))
    return model, ckpt


def generate_actions(model, rows, device, total_steps=360):
    # seed with first true sequence, then free-run.
    seq = [list(map(int, a)) for a in rows[0]["seq"]]
    actions = []
    with torch.no_grad():
        for i in range(total_steps):
            x = torch.tensor([seq[-8:]], dtype=torch.float32, device=device) / 4.0
            pred = model(x).argmax(dim=-1)[0].cpu().tolist()
            frame = i * 5
            pairs = [[j, int(v)] for j, v in enumerate(pred) if int(v) != 0]
            actions.append({"frame": frame, "values": pred, "pairs": pairs})
            seq.append(pred)
    return actions


def write_action_only_rpl(actions):
    # Use header from template but write action-only continuation for quick visual test.
    header = []
    for line in TEMPLATE.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("FRAME "):
            break
        if line.startswith("FIGHTNAME 0;"):
            header.append(f"FIGHTNAME 0; {OUT.stem}")
        elif line.startswith("NEWGAME 0;"):
            parts = line.split(";")
            if len(parts) > 1:
                nums = parts[1].split()
                if nums:
                    nums[0] = "2200"
                header.append("NEWGAME 0;" + " ".join(nums))
            else:
                header.append(line)
        else:
            header.append(line)
    out = header + [""]
    for a in actions:
        out.append(f"FRAME {a['frame']}; 0 0 0 0")
        if a["pairs"]:
            flat = " ".join(f"{j} {v}" for j, v in a["pairs"])
            out.append(f"JOINT 0; {flat}")
        out.append("")
    OUT.write_text("\n".join(out) + "\n", encoding="utf-8")


def main():
    if not MODEL.exists():
        raise FileNotFoundError(f"Missing model: {MODEL}\nRun train_xioi_loop_len265_gru_v52.py first.")
    rows = load_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(device)
    actions = generate_actions(model, rows, device)
    ACTIONS_JSON.write_text(json.dumps({"version": 52, "actions": actions}, indent=2), encoding="utf-8")
    write_action_only_rpl(actions)

    for d in (REPLAY_ROOT, REPLAY_PARKOUR):
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT, d / OUT.name)
        print("Copied to:", d / OUT.name)
    counts = Counter(v for a in actions for v in a["values"])
    print("Wrote:", OUT)
    print("Actions:", ACTIONS_JSON)
    print("Pred counts:", counts.most_common())


if __name__ == "__main__":
    main()
