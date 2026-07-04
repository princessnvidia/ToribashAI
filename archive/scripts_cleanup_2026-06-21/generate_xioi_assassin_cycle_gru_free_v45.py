#!/usr/bin/env python3
"""
generate_xioi_assassin_cycle_gru_free_v45.py

Generate a long RPL by copying the source template up to frame 315, then adding
free-running GRU cycle actions after 315. This version uses the V45 real-action
cycle GRU, not the broken V44 all-zero dataset.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET = ROOT / "datasets" / "ml" / "xioi_assassin_cycle_v45_sequences.jsonl"
MODEL_PATH = ROOT / "models" / "xioi_assassin_cycle_gru_v45.pt"
TEMPLATE_RPL = ROOT / "generated_replays" / "xioi_427_assassincreedhunter_v37.rpl"
OUT_RPL = ROOT / "generated_replays" / "xioi_assassin_cycle_gru_free_v45.rpl"
OUT_ACTIONS = ROOT / "generated_replays" / "xioi_assassin_cycle_gru_free_v45_actions.json"
OUT_SUMMARY = ROOT / "generated_replays" / "xioi_assassin_cycle_gru_free_v45_summary.json"
STEAM_REPLAY = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
STEAM_PARKOUR = STEAM_REPLAY / "parkour"

SOURCE_UNTIL = 315
GENERATE_UNTIL = 1200
TURNFRAMES = 5
ACTION_DIM = 20


class GRUAction(nn.Module):
    def __init__(self, state_dim=20, hidden=192, layers=2):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden, num_layers=layers, batch_first=True, dropout=0.05 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, ACTION_DIM * 5))

    def forward(self, x):
        out, _ = self.gru(x)
        z = out[:, -1]
        return self.head(z).view(-1, ACTION_DIM, 5)


def load_rows():
    if not DATASET.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET}\nRun build_xioi_assassin_cycle_dataset_v45.py first.")
    return [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]


def load_model(device):
    ckpt = torch.load(MODEL_PATH, map_location=device)
    hidden = int(ckpt.get("hidden", 192))
    layers = int(ckpt.get("layers", 2))
    state_dim = int(ckpt.get("state_dim", 20))
    model = GRUAction(state_dim=state_dim, hidden=hidden, layers=layers).to(device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    print("Loaded:", MODEL_PATH, "hidden=", hidden, "layers=", layers, "epoch=", ckpt.get("epoch"))
    return model, state_dim


def action_to_pairs(action):
    return [[j, int(v)] for j, v in enumerate(action) if int(v) != 0]


def generate_actions(model, state_dim, seed_seq, device):
    seq = [list(map(int, a)) for a in seed_seq]
    generated = []
    frame = SOURCE_UNTIL + TURNFRAMES
    with torch.no_grad():
        while frame <= GENERATE_UNTIL:
            x = torch.tensor([seq[-8:]], dtype=torch.float32, device=device) / 4.0
            logits = model(x)
            pred = logits.argmax(dim=-1)[0].cpu().tolist()
            pred = [int(v) for v in pred]
            generated.append({"frame": frame, "action": pred, "pairs": action_to_pairs(pred)})
            seq.append(pred)
            frame += TURNFRAMES
    return generated


def strip_after_source_until(lines):
    out = []
    keep = True
    for line in lines:
        if line.startswith("FIGHTNAME 0;"):
            out.append("FIGHTNAME 0; xioi_assassin_cycle_gru_free_v45")
            continue
        if line.startswith("FRAME "):
            try:
                fr = int(line.split()[1].split(";")[0])
                keep = fr <= SOURCE_UNTIL
            except Exception:
                keep = True
        if keep:
            out.append(line)
    return out


def append_generated(lines, generated):
    out = list(lines)
    for g in generated:
        out.append("")
        out.append(f"FRAME {g['frame']}; 0 0 0 0")
        pairs = g["pairs"]
        if pairs:
            flat = " ".join(f"{j} {v}" for j, v in pairs)
            out.append(f"JOINT 0; {flat}")
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    rows = load_rows()
    model, state_dim = load_model(device)

    # Use the first real cycle seed, not zeros.
    seed_seq = rows[0]["seq"]
    generated = generate_actions(model, state_dim, seed_seq, device)

    lines = TEMPLATE_RPL.read_text(encoding="utf-8", errors="ignore").splitlines()
    base = strip_after_source_until(lines)
    out = append_generated(base, generated)

    OUT_RPL.parent.mkdir(parents=True, exist_ok=True)
    OUT_RPL.write_text("\n".join(out) + "\n", encoding="utf-8")
    OUT_ACTIONS.write_text(json.dumps({"version": 45, "actions": generated}, indent=2), encoding="utf-8")

    STEAM_REPLAY.mkdir(parents=True, exist_ok=True)
    STEAM_PARKOUR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_RPL, STEAM_REPLAY / OUT_RPL.name)
    shutil.copy2(OUT_RPL, STEAM_PARKOUR / OUT_RPL.name)

    from collections import Counter
    cnt = Counter()
    active = Counter()
    for g in generated:
        cnt.update(g["action"])
        active[len(g["pairs"])] += 1
    summary = {
        "version": 45,
        "rpl": str(OUT_RPL),
        "model": str(MODEL_PATH),
        "dataset": str(DATASET),
        "source_until": SOURCE_UNTIL,
        "generate_until": GENERATE_UNTIL,
        "generated_frames": len(generated),
        "pred_counts": cnt.most_common(),
        "active_distribution": active.most_common(),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote:", OUT_RPL)
    print("Copied to:", STEAM_REPLAY / OUT_RPL.name)
    print("Copied to:", STEAM_PARKOUR / OUT_RPL.name)
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
