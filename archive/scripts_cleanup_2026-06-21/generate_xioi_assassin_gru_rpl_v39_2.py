#!/usr/bin/env python3
"""
generate_xioi_assassin_gru_rpl_v39_2.py

V39.2 = template-safe GRU validation export.

Goal:
  - Load the V38 GRU checkpoint.
  - Verify predictions against the V38 dataset.
  - Write a replay by copying the original template RPL almost verbatim.
  - DO NOT reconstruct FRAME/POS/QAT/LINVEL/ANGVEL blocks.
  - Preserve the source replay timing/physics exactly.
  - Only update FIGHTNAME and optionally write predicted JOINT lines for frames
    where the dataset has a prediction.

Important:
  If predictions are exact, this should be visually identical to:
    generated_replays/xioi_427_assassincreedhunter_v37.rpl

This avoids the V39.1 bug where the replay was rebuilt from sequence rows and
some JOINT blocks around frame ~300 were shifted/replaced incorrectly.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET = ROOT / "datasets/ml/xioi_assassin_walk_v38_sequences.jsonl"
MODEL_PATH = ROOT / "models/xioi_assassin_gru_v38.pt"
TEMPLATE_RPL = ROOT / "generated_replays/xioi_427_assassincreedhunter_v37.rpl"
OUT_DIR = ROOT / "generated_replays"
OUT_RPL = OUT_DIR / "xioi_assassin_gru_generated_v39_2_template_safe.rpl"
OUT_ACTIONS = OUT_DIR / "xioi_assassin_gru_generated_v39_2_actions.json"
OUT_SUMMARY = OUT_DIR / "xioi_assassin_gru_generated_v39_2_summary.json"

STEAM_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)
STEAM_PARKOUR_DIR = STEAM_REPLAY_DIR / "parkour"

RPL_FIGHTNAME = "xioi_assassin_gru_generated_v39_2_template_safe"


class GRUAction(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 160, layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
        )
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 20 * 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        z = y[:, -1, :]
        return self.head(z).view(-1, 20, 5)


def load_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"Dataset empty: {DATASET}")
    return rows


def row_state(row: dict[str, Any]) -> list[list[float]]:
    for key in ("states", "state_seq", "sequence", "x"):
        if key in row:
            return row[key]
    raise KeyError(f"Cannot find state sequence key in row keys={list(row.keys())}")


def row_action(row: dict[str, Any]) -> list[int]:
    for key in ("action", "target", "y"):
        if key in row:
            return [int(v) for v in row[key]]
    raise KeyError(f"Cannot find action key in row keys={list(row.keys())}")


def row_frame(row: dict[str, Any], fallback: int) -> int:
    for key in ("target_frame", "frame", "next_frame", "action_frame"):
        if key in row:
            return int(row[key])
    return fallback


def load_model(state_dim: int, device: torch.device) -> GRUAction:
    ckpt = torch.load(MODEL_PATH, map_location=device)
    hidden = int(ckpt.get("hidden", 160)) if isinstance(ckpt, dict) else 160
    layers = int(ckpt.get("layers", 2)) if isinstance(ckpt, dict) else 2
    model = GRUAction(state_dim=state_dim, hidden=hidden, layers=layers).to(device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded model: {MODEL_PATH} hidden={hidden} layers={layers} epoch={ckpt.get('epoch') if isinstance(ckpt, dict) else None}")
    return model


def predict(rows: list[dict[str, Any]], model: GRUAction, device: torch.device) -> tuple[dict[int, list[int]], dict[str, Any]]:
    predicted_by_frame: dict[int, list[int]] = {}
    true_by_frame: dict[int, list[int]] = {}

    true_counts: Counter[int] = Counter()
    pred_counts: Counter[int] = Counter()
    total = 0
    correct = 0
    exact = 0

    with torch.no_grad():
        for idx, row in enumerate(rows):
            states = row_state(row)
            true_action = row_action(row)
            frame = row_frame(row, fallback=idx * 5)
            x = torch.tensor([states], dtype=torch.float32, device=device)
            logits = model(x)[0]
            pred_action = torch.argmax(logits, dim=-1).cpu().tolist()
            pred_action = [int(v) for v in pred_action]

            predicted_by_frame[frame] = pred_action
            true_by_frame[frame] = true_action

            if pred_action == true_action:
                exact += 1
            for p, t in zip(pred_action, true_action):
                pred_counts[p] += 1
                true_counts[t] += 1
                total += 1
                if p == t:
                    correct += 1

    summary = {
        "rows_used": len(rows),
        "joint_accuracy_on_export": correct / max(1, total),
        "exact_accuracy_on_export": exact / max(1, len(rows)),
        "true_counts": true_counts.most_common(),
        "pred_counts": pred_counts.most_common(),
        "frames_predicted": sorted(predicted_by_frame),
        "frame_min": min(predicted_by_frame) if predicted_by_frame else None,
        "frame_max": max(predicted_by_frame) if predicted_by_frame else None,
    }

    actions_dump = {
        "name": "xioi_assassin_gru_generated_v39_2_actions",
        "version": "39.2",
        "mode": "template_safe_predictions",
        "actions": [
            {
                "frame": fr,
                "values": predicted_by_frame[fr],
                "pairs": [[j, v] for j, v in enumerate(predicted_by_frame[fr]) if int(v) != 0],
                "true_values": true_by_frame.get(fr),
            }
            for fr in sorted(predicted_by_frame)
        ],
    }
    OUT_ACTIONS.write_text(json.dumps(actions_dump, indent=2), encoding="utf-8")
    return predicted_by_frame, summary


def compact_joint_line(player: int, action: list[int]) -> str:
    pairs: list[str] = []
    for j, v in enumerate(action):
        if int(v) != 0:
            pairs.append(f"{j} {int(v)}")
    if not pairs:
        return f"JOINT {player};"
    return f"JOINT {player}; " + " ".join(pairs)


def parse_frame_number(line: str) -> int | None:
    m = re.match(r"^FRAME\s+(-?\d+)\s*;", line.strip())
    return int(m.group(1)) if m else None


def is_joint0_line(line: str) -> bool:
    return line.lstrip().startswith("JOINT 0;")


def is_joint_line(line: str) -> bool:
    return re.match(r"^\s*JOINT\s+\d+\s*;", line) is not None


def rewrite_template_rpl(predicted_by_frame: dict[int, list[int]]) -> dict[str, Any]:
    if not TEMPLATE_RPL.exists():
        raise FileNotFoundError(TEMPLATE_RPL)

    lines = TEMPLATE_RPL.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[str] = []

    current_frame: int | None = None
    replaced_frames: set[int] = set()
    removed_joint0_lines = 0
    inserted_joint0_lines = 0

    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("FIGHTNAME 0;"):
            out.append(f"FIGHTNAME 0; {RPL_FIGHTNAME}")
            i += 1
            continue

        frame_no = parse_frame_number(line)
        if frame_no is not None:
            current_frame = frame_no
            out.append(line)
            i += 1

            # If this frame has a GRU prediction, remove all immediate JOINT 0 lines
            # in this frame block and insert exactly one compact JOINT line.
            if current_frame in predicted_by_frame:
                action = predicted_by_frame[current_frame]
                while i < len(lines):
                    nxt = lines[i]
                    if parse_frame_number(nxt) is not None:
                        break
                    # Preserve comments and physics lines, but skip player-0 JOINTs.
                    if is_joint0_line(nxt):
                        removed_joint0_lines += 1
                        i += 1
                        continue
                    out.append(nxt)
                    i += 1
                out.append(compact_joint_line(0, action))
                replaced_frames.add(current_frame)
                inserted_joint0_lines += 1
            continue

        # If template has JOINT 0 outside a parsed frame, leave it alone.
        out.append(line)
        i += 1

    # Do not add missing predicted frames. Template-safety means no new FRAME blocks.
    # Missing frames are reported only.
    missing_predicted_frames = sorted(set(predicted_by_frame) - replaced_frames)

    OUT_RPL.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {
        "template_lines": len(lines),
        "output_lines": len(out),
        "predicted_frames": len(predicted_by_frame),
        "replaced_frames": len(replaced_frames),
        "missing_predicted_frames": missing_predicted_frames,
        "removed_joint0_lines": removed_joint0_lines,
        "inserted_joint0_lines": inserted_joint0_lines,
    }


def copy_to_steam() -> list[str]:
    copied: list[str] = []
    for dst_dir in (STEAM_REPLAY_DIR, STEAM_PARKOUR_DIR):
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / OUT_RPL.name
        shutil.copy2(OUT_RPL, dst)
        copied.append(str(dst))
    return copied


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    rows = load_rows()
    states0 = row_state(rows[0])
    state_dim = len(states0[0])
    print("Rows:", len(rows), "state_dim:", state_dim)

    model = load_model(state_dim, device)
    predicted_by_frame, pred_summary = predict(rows, model, device)
    rewrite_summary = rewrite_template_rpl(predicted_by_frame)
    copied = copy_to_steam()

    summary = {
        "version": "39.2",
        "mode": "template_safe_gru_to_rpl",
        "note": "Copies the source template RPL and replaces only JOINT 0 lines inside existing FRAME blocks.",
        **pred_summary,
        "rewrite": rewrite_summary,
        "model": str(MODEL_PATH),
        "dataset": str(DATASET),
        "template": str(TEMPLATE_RPL),
        "rpl": str(OUT_RPL),
        "actions_json": str(OUT_ACTIONS),
        "copied_to": copied,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Wrote:", OUT_RPL)
    for dst in copied:
        print("Copied to:", dst)
    print("Summary:")
    print(json.dumps({
        "joint_accuracy_on_export": summary["joint_accuracy_on_export"],
        "exact_accuracy_on_export": summary["exact_accuracy_on_export"],
        "rewrite": rewrite_summary,
        "rpl": str(OUT_RPL),
    }, indent=2))


if __name__ == "__main__":
    main()
