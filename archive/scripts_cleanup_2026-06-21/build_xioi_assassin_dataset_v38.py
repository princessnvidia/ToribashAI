#!/usr/bin/env python3
"""
build_xioi_assassin_dataset_v38.py

Build a compact GRU imitation dataset from the V38 Xioi assassincreedhunter
reference replay.

Rows are sequence -> next action targets, using body point positions + previous
joint values. This is intentionally small and pure: 100% one validated walk.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
REF = ROOT / "generated_replays" / "xioi_assassin_reference_v38.json"
OUT = ROOT / "datasets/ml/xioi_assassin_walk_v38_sequences.jsonl"
SUMMARY = ROOT / "generated_replays/xioi_assassin_walk_v38_dataset_summary.json"
SEQ_LEN = 8

POINT_ORDER = ["head", "chest", "lumbar", "abs", "left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_foot", "right_foot"]


def zeros(n: int) -> list[float]:
    return [0.0] * n


def normalize_point(pt: list[float] | None, origin: list[float] | None) -> list[float]:
    if not pt or len(pt) < 3:
        return [0.0, 0.0, 0.0]
    if origin and len(origin) >= 3:
        return [float(pt[0] - origin[0]), float(pt[1] - origin[1]), float(pt[2] - origin[2])]
    return [float(pt[0]), float(pt[1]), float(pt[2])]


def action_vector(pairs: list[list[int]]) -> list[int]:
    arr = [0] * 20
    for pair in pairs or []:
        if len(pair) != 2:
            continue
        j, v = int(pair[0]), int(pair[1])
        if 0 <= j < 20 and 0 <= v <= 4:
            arr[j] = v
    return arr


def main() -> None:
    if not REF.exists():
        raise FileNotFoundError(f"Missing {REF}. Run build_xioi_assassin_reference_v38.py first.")
    data = json.loads(REF.read_text(encoding="utf-8"))
    rows = data.get("walking_reference", [])
    rows = sorted(rows, key=lambda r: int(r["frame"]))
    if len(rows) <= SEQ_LEN:
        raise RuntimeError("Not enough rows for sequence dataset")

    first_chest = None
    for r in rows:
        first_chest = r.get("points", {}).get("chest")
        if first_chest:
            break

    states: list[list[float]] = []
    actions: list[list[int]] = []
    frames: list[int] = []
    prev_action = [0] * 20

    for r in rows:
        pts = r.get("points", {})
        state: list[float] = []
        for name in POINT_ORDER:
            state.extend(normalize_point(pts.get(name), first_chest))
        state.extend(float(x) / 4.0 for x in prev_action)
        act = action_vector(r.get("joint_pairs", []))
        states.append(state)
        actions.append(act)
        frames.append(int(r["frame"]))
        prev_action = act

    OUT.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    value_counts = Counter()
    with OUT.open("w", encoding="utf-8") as f:
        for i in range(0, len(states) - SEQ_LEN):
            seq = states[i : i + SEQ_LEN]
            target = actions[i + SEQ_LEN]
            for v in target:
                value_counts[v] += 1
            row = {
                "source": str(REF),
                "seq_len": SEQ_LEN,
                "frame": frames[i + SEQ_LEN],
                "state_seq": seq,
                "action": target,
            }
            f.write(json.dumps(row) + "\n")
            count += 1

    summary = {
        "version": 38,
        "source": str(REF),
        "dataset": str(OUT),
        "rows": count,
        "seq_len": SEQ_LEN,
        "state_dim": len(states[0]),
        "action_dim": 20,
        "frame_min": min(frames),
        "frame_max": max(frames),
        "value_counts": value_counts.most_common(),
        "point_order": POINT_ORDER,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Dataset:", OUT)
    print("Summary:", SUMMARY)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
