#!/usr/bin/env python3
"""
Generate V44 free-running loop RPL.

Replay structure:
  - copy source/template safely
  - use true/source movement until frame 315
  - after frame 315, write generated JOINT 0 commands on existing/extra FRAME blocks

This tests whether the GRU learned the explicit 70->295 cycle transition.
"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter, deque
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET = ROOT / "datasets" / "ml" / "xioi_assassin_cycle_v44_sequences.jsonl"
MODEL_PATH = ROOT / "models" / "xioi_assassin_cycle_gru_v44.pt"
TEMPLATE_RPL = ROOT / "generated_replays" / "xioi_427_assassincreedhunter_v37.rpl"
OUT_DIR = ROOT / "generated_replays"
STEAM_REPLAY = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
STEAM_PARKOUR = STEAM_REPLAY / "parkour"

OUT_RPL = OUT_DIR / "xioi_assassin_cycle_gru_free_v44.rpl"
OUT_ACTIONS = OUT_DIR / "xioi_assassin_cycle_gru_free_v44_actions.json"
OUT_SUMMARY = OUT_DIR / "xioi_assassin_cycle_gru_free_v44_summary.json"

SEQ_LEN = 8
CYCLE_START = 70
CYCLE_END = 295
SOURCE_UNTIL = 315
GENERATE_UNTIL = 1200
TURN_STEP = 5

POINT_ORDER = [
    "head", "chest", "lumbar", "abs",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_foot", "right_foot",
]


class GRUAction(nn.Module):
    def __init__(self, state_dim: int, hidden: int, layers: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.10 if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 20 * 5))

    def forward(self, x):
        y, _ = self.gru(x)
        return self.head(y[:, -1, :]).view(-1, 20, 5)


def load_rows() -> list[dict[str, Any]]:
    if not DATASET.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET}\nRun build_xioi_assassin_cycle_dataset_v44.py first.")
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_model(state_dim: int, device):
    ckpt = torch.load(MODEL_PATH, map_location=device)
    hidden = int(ckpt.get("hidden", 192))
    layers = int(ckpt.get("layers", 2))
    model = GRUAction(state_dim, hidden, layers).to(device)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    print("Loaded:", MODEL_PATH, "hidden=", hidden, "layers=", layers, "epoch=", ckpt.get("epoch"))
    return model


def action_to_pairs(action: list[int]) -> list[list[int]]:
    return [[j, int(v)] for j, v in enumerate(action) if int(v) != 0]


def pairs_to_action(pairs: list[list[int]]) -> list[int]:
    a = [0] * 20
    for j, v in pairs:
        if 0 <= int(j) < 20:
            a[int(j)] = int(v)
    return a


def parse_frame_blocks(lines: list[str]) -> dict[int, tuple[int, int]]:
    blocks = {}
    starts = []
    for i, line in enumerate(lines):
        m = re.match(r"^FRAME\s+(\d+);", line)
        if m:
            starts.append((int(m.group(1)), i))
    for idx, (fr, start) in enumerate(starts):
        end = starts[idx + 1][1] if idx + 1 < len(starts) else len(lines)
        blocks[fr] = (start, end)
    return blocks


def remove_joint0_and_insert(lines: list[str], frame_actions: dict[int, list[int]], fightname: str) -> list[str]:
    out = list(lines)
    for i, line in enumerate(out):
        if line.startswith("FIGHTNAME 0;"):
            out[i] = f"FIGHTNAME 0; {fightname}"
            break

    blocks = parse_frame_blocks(out)
    # Update existing blocks up to template end.
    new_lines = []
    i = 0
    while i < len(out):
        m = re.match(r"^FRAME\s+(\d+);", out[i])
        if not m:
            new_lines.append(out[i])
            i += 1
            continue
        fr = int(m.group(1))
        start, end = blocks[fr]
        block = out[start:end]
        block2 = [line for line in block if not line.startswith("JOINT 0;")]
        if fr in frame_actions:
            pairs = action_to_pairs(frame_actions[fr])
            if pairs:
                flat = " ".join(f"{j} {v}" for j, v in pairs)
                # Insert after FRAME line.
                block2.insert(1, f"JOINT 0; {flat}")
        new_lines.extend(block2)
        i = end

    # Append generated frames beyond template.
    existing = set(blocks)
    extra_frames = [fr for fr in sorted(frame_actions) if fr not in existing]
    for fr in extra_frames:
        pairs = action_to_pairs(frame_actions[fr])
        new_lines.append("")
        new_lines.append(f"FRAME {fr}; 0 0 0 0")
        if pairs:
            flat = " ".join(f"{j} {v}" for j, v in pairs)
            new_lines.append(f"JOINT 0; {flat}")
    return new_lines


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = load_rows()
    state_dim = len(rows[0]["states"][0])
    print("Device:", device)
    print("Rows:", len(rows), "state_dim:", state_dim)
    model = load_model(state_dim, device)

    # Seed with first seq from cycle dataset (already cycle-only 70->...).
    seq = deque(rows[0]["states"], maxlen=SEQ_LEN)

    # Map original cycle frame -> true action from dataset for source-like first cycle.
    by_orig: dict[int, list[int]] = {}
    for r in rows:
        fr = int(r.get("target_original_frame", -1))
        if fr not in by_orig:
            by_orig[fr] = [int(x) for x in r["action"]]

    generated: dict[int, list[int]] = {}
    pred_counts = Counter()

    # Keep source action until SOURCE_UNTIL when available, then generate cyclic continuation.
    # For output after 315, generate at 5-frame intervals.
    with torch.no_grad():
        for out_frame in range(SOURCE_UNTIL + TURN_STEP, GENERATE_UNTIL + 1, TURN_STEP):
            x = torch.tensor([list(seq)], dtype=torch.float32, device=device)
            logits = model(x)
            action = logits.argmax(dim=-1)[0].cpu().tolist()
            generated[out_frame] = [int(v) for v in action]
            pred_counts.update(action)

            # Advance approximate state by cycling dataset states. This keeps phase moving through learned loop.
            # Use generated step index mapped to rows states to avoid impossible all-zero state drift.
            idx = (len(generated) + SEQ_LEN) % len(rows)
            next_state = rows[idx]["states"][-1]
            seq.append(next_state)

    template_lines = TEMPLATE_RPL.read_text(encoding="utf-8", errors="ignore").splitlines()
    fightname = OUT_RPL.stem
    output_lines = remove_joint0_and_insert(template_lines, generated, fightname)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RPL.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

    STEAM_REPLAY.mkdir(parents=True, exist_ok=True)
    STEAM_PARKOUR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_RPL, STEAM_REPLAY / OUT_RPL.name)
    shutil.copy2(OUT_RPL, STEAM_PARKOUR / OUT_RPL.name)

    actions_json = {
        "version": 44,
        "mode": "free_cycle_after_315",
        "source_until": SOURCE_UNTIL,
        "generate_until": GENERATE_UNTIL,
        "actions": [
            {"frame": fr, "values": generated[fr], "pairs": action_to_pairs(generated[fr])}
            for fr in sorted(generated)
        ],
    }
    OUT_ACTIONS.write_text(json.dumps(actions_json, indent=2), encoding="utf-8")
    summary = {
        "version": 44,
        "rpl": str(OUT_RPL),
        "model": str(MODEL_PATH),
        "dataset": str(DATASET),
        "source_until": SOURCE_UNTIL,
        "generate_until": GENERATE_UNTIL,
        "generated_frames": len(generated),
        "pred_counts": pred_counts.most_common(),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Wrote:", OUT_RPL)
    print("Copied to:", STEAM_REPLAY / OUT_RPL.name)
    print("Copied to:", STEAM_PARKOUR / OUT_RPL.name)
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
