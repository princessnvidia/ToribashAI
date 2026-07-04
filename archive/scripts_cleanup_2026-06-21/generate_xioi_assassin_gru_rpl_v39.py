#!/usr/bin/env python3
"""
generate_xioi_assassin_gru_rpl_v39.py

V39: use the overfitted V38 GRU to regenerate the Xioi assassincreedhunter walk as an RPL.

Inputs:
  models/xioi_assassin_gru_v38.pt
  datasets/ml/xioi_assassin_walk_v38_sequences.jsonl
  generated_replays/xioi_427_assassincreedhunter_v37.rpl

Outputs:
  generated_replays/xioi_assassin_gru_generated_v39.rpl
  generated_replays/xioi_assassin_gru_generated_v39_actions.json
  generated_replays/xioi_assassin_gru_generated_v39_summary.json

Important: this exports a full RPL by taking the V37 RPL as a physical/template base
and replacing JOINT commands with the GRU-predicted actions. POS/QAT/LINVEL/ANGVEL
are kept from the template to preserve replay context while comparing actions.
"""

from __future__ import annotations

import json
import random
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
OUT_RPL = OUT_DIR / "xioi_assassin_gru_generated_v39.rpl"
OUT_ACTIONS = OUT_DIR / "xioi_assassin_gru_generated_v39_actions.json"
OUT_SUMMARY = OUT_DIR / "xioi_assassin_gru_generated_v39_summary.json"
STEAM_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)
STEAM_PARKOUR_DIR = STEAM_REPLAY_DIR / "parkour"

SEQ_LEN = 8
ACTION_DIM = 20
NUM_CLASSES = 5
GENERATE_STEPS = 93  # V38 reference frame count
TURNFRAMES = 5
FIGHTNAME = "xioi_assassin_gru_generated_v39"


class GRUAction(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128, layers: int = 1):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden, layers, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, ACTION_DIM * NUM_CLASSES))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        z = y[:, -1, :]
        return self.head(z).view(-1, ACTION_DIM, NUM_CLASSES)


def load_rows() -> list[dict[str, Any]]:
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)
    rows = []
    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty dataset: {DATASET}")
    return rows


def infer_state_dim(row: dict[str, Any]) -> int:
    seq = row.get("state_seq") or row.get("states") or row.get("x")
    if not seq:
        raise KeyError("dataset row has no state_seq/states/x")
    return len(seq[0])


def get_seq(row: dict[str, Any]) -> list[list[float]]:
    seq = row.get("state_seq") or row.get("states") or row.get("x")
    if not seq:
        raise KeyError("dataset row has no state sequence")
    return seq


def get_target(row: dict[str, Any]) -> list[int]:
    y = row.get("action") or row.get("target") or row.get("y")
    if y is None:
        # Some builders store target_action
        y = row.get("target_action")
    if y is None:
        raise KeyError("dataset row has no action/target/y")
    if isinstance(y, dict):
        arr = [0] * ACTION_DIM
        for k, v in y.items():
            arr[int(k)] = int(v)
        return arr
    return [int(v) for v in y]


def load_model(state_dim: int, device: torch.device) -> nn.Module:
    raw = torch.load(MODEL_PATH, map_location=device)
    state = raw.get("model_state_dict") or raw.get("state_dict") or raw
    hidden = 128
    # If checkpoint has metadata, trust it.
    if isinstance(raw, dict):
        hidden = int(raw.get("hidden", raw.get("HIDDEN", hidden)))
    # Infer hidden from GRU weight if needed.
    for k, v in state.items():
        if k.endswith("weight_hh_l0"):
            hidden = int(v.shape[1])
            break
    model = GRUAction(state_dim=state_dim, hidden=hidden).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict_teacher_forced(rows: list[dict[str, Any]], model: nn.Module, device: torch.device) -> list[dict[str, Any]]:
    """Predict one action for each dataset sequence. This is teacher-forced: every prediction
    sees real reference state_seq, so an overfitted model should reproduce the action list.
    """
    actions = []
    true_counter = Counter()
    pred_counter = Counter()
    joint_correct = 0
    joint_total = 0
    exact = 0

    for i, row in enumerate(rows[:GENERATE_STEPS]):
        seq = torch.tensor([get_seq(row)], dtype=torch.float32, device=device)
        true = get_target(row)
        with torch.no_grad():
            logits = model(seq)[0]
            pred = logits.argmax(dim=-1).cpu().tolist()

        pairs = [[j, int(v)] for j, v in enumerate(pred) if int(v) != 0]
        frame = int(row.get("frame", row.get("target_frame", i * TURNFRAMES)))
        # normalize generated frame spacing for RPL readability
        out_frame = i * TURNFRAMES
        actions.append({
            "frame": out_frame,
            "source_frame": frame,
            "pairs": pairs,
            "values": pred,
            "true_values": true,
        })
        for a, b in zip(pred, true):
            pred_counter[int(a)] += 1
            true_counter[int(b)] += 1
            joint_correct += int(int(a) == int(b))
            joint_total += 1
        exact += int(pred == true)

    summary = {
        "version": 39,
        "mode": "teacher_forced_gru_to_rpl",
        "rows_used": len(actions),
        "joint_accuracy_on_export": joint_correct / max(1, joint_total),
        "exact_accuracy_on_export": exact / max(1, len(actions)),
        "true_counts": true_counter.most_common(),
        "pred_counts": pred_counter.most_common(),
    }
    return actions, summary


def parse_frame_no(line: str) -> int | None:
    m = re.match(r"^FRAME\s+(\d+)\s*;", line)
    return int(m.group(1)) if m else None


def rewrite_rpl_with_actions(actions: list[dict[str, Any]]) -> list[str]:
    if not TEMPLATE_RPL.exists():
        raise FileNotFoundError(TEMPLATE_RPL)
    action_by_frame = {int(a["frame"]): a.get("pairs", []) for a in actions}
    lines = TEMPLATE_RPL.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[str] = []
    current_frame: int | None = None
    wrote_for_frame: set[int] = set()

    for line in lines:
        fr = parse_frame_no(line)
        if fr is not None:
            current_frame = fr
            out.append(line)
            if fr in action_by_frame:
                out.append(f"# V39 GRU generated action for frame {fr}")
                for j, v in action_by_frame[fr]:
                    out.append(f"JOINT 0; {int(j)} {int(v)}")
                wrote_for_frame.add(fr)
            continue

        if line.startswith("FIGHTNAME 0;"):
            out.append(f"FIGHTNAME 0; {FIGHTNAME}")
            continue

        # remove old JOINT lines on frames we regenerate; keep POS/QAT/vel/context.
        if current_frame in action_by_frame and line.startswith("JOINT 0;"):
            continue

        out.append(line)

    # Append missing action frames if template didn't have every normalized frame.
    missing = sorted(set(action_by_frame) - wrote_for_frame)
    for fr in missing:
        out.append("")
        out.append(f"FRAME {fr};")
        out.append(f"# V39 appended GRU generated action for frame {fr}")
        for j, v in action_by_frame[fr]:
            out.append(f"JOINT 0; {int(j)} {int(v)}")

    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    rows = load_rows()
    state_dim = infer_state_dim(rows[0])
    print("Rows:", len(rows), "state_dim:", state_dim)
    model = load_model(state_dim, device)
    actions, summary = predict_teacher_forced(rows, model, device)
    OUT_ACTIONS.write_text(json.dumps({"version": 39, "actions": actions}, indent=2), encoding="utf-8")
    rpl_lines = rewrite_rpl_with_actions(actions)
    OUT_RPL.write_text("\n".join(rpl_lines) + "\n", encoding="utf-8")
    summary.update({
        "model": str(MODEL_PATH),
        "dataset": str(DATASET),
        "template": str(TEMPLATE_RPL),
        "rpl": str(OUT_RPL),
        "actions_json": str(OUT_ACTIONS),
    })
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for d in [STEAM_REPLAY_DIR, STEAM_PARKOUR_DIR]:
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT_RPL, d / OUT_RPL.name)
        print("Copied to:", d / OUT_RPL.name)

    print("Wrote:", OUT_RPL)
    print("Summary:", json.dumps(summary, indent=2)[:2000])


if __name__ == "__main__":
    main()
