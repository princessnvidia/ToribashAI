#!/usr/bin/env python3
"""
build_xioi_assassin_cycle_dataset_v44.py

V44 = explicit cycle dataset.
We keep the useful Xioi walking cycle only, then repeat it several times:
    70 -> 295 -> 70 -> 295 -> ...

Goal: teach the GRU the missing transition 295 -> 70.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
REF_PATH = ROOT / "generated_replays" / "xioi_assassin_reference_v43.json"
OUT_DATASET = ROOT / "datasets" / "ml" / "xioi_assassin_cycle_v44_sequences.jsonl"
OUT_SUMMARY = ROOT / "generated_replays" / "xioi_assassin_cycle_v44_dataset_summary.json"

SEQ_LEN = 8
CYCLE_START = 70
CYCLE_END = 295
CYCLES = 8

# Keep same point order as v38/v43, fallback if missing.
POINT_ORDER = [
    "head", "chest", "lumbar", "abs",
    "left_shoulder", "right_shoulder",
    "left_hip", "right_hip",
    "left_foot", "right_foot",
]


def load_reference() -> dict[str, Any]:
    if not REF_PATH.exists():
        raise FileNotFoundError(f"Missing reference: {REF_PATH}\nRun build_xioi_assassin_reference_v43.py first.")
    return json.loads(REF_PATH.read_text(encoding="utf-8"))


def get_frames(ref: dict[str, Any]) -> list[dict[str, Any]]:
    frames = ref.get("frames") or ref.get("reference") or []
    if isinstance(frames, dict):
        frames = [dict(v, frame=int(k)) for k, v in sorted(frames.items(), key=lambda kv: int(kv[0]))]
    frames = [f for f in frames if CYCLE_START <= int(f.get("frame", -999999)) <= CYCLE_END]
    frames.sort(key=lambda f: int(f.get("frame", 0)))
    if len(frames) < SEQ_LEN + 2:
        raise RuntimeError(f"Not enough cycle frames in {REF_PATH}: {len(frames)}")
    return frames


def point_xyz(frame: dict[str, Any], name: str) -> list[float]:
    points = frame.get("points") or frame.get("body") or {}
    v = points.get(name)
    if v is None:
        # support flat fields like frame['head'] = [x,y,z]
        v = frame.get(name, [0.0, 0.0, 0.0])
    return [float(v[0]), float(v[1]), float(v[2])]


def state_from_frame(frame: dict[str, Any]) -> list[float]:
    vals: list[float] = []
    for p in POINT_ORDER:
        vals.extend(point_xyz(frame, p))
    # Add compact phase features to help the GRU know where it is in the loop.
    fr = int(frame.get("frame", 0))
    phase = (fr - CYCLE_START) / max(1, CYCLE_END - CYCLE_START)
    vals.append(float(phase))
    vals.append(float(1.0 - phase))
    return vals


def action_from_frame(frame: dict[str, Any]) -> list[int]:
    action = [0] * 20
    joints = frame.get("joints") or {}
    pairs = frame.get("joint_pairs") or frame.get("pairs") or []
    if isinstance(joints, dict) and joints:
        for k, v in joints.items():
            j = int(k)
            if 0 <= j < 20:
                action[j] = int(v)
    else:
        for p in pairs:
            if len(p) >= 2:
                j, v = int(p[0]), int(p[1])
                if 0 <= j < 20:
                    action[j] = int(v)
    return action


def main() -> None:
    ref = load_reference()
    cycle = get_frames(ref)

    timeline: list[dict[str, Any]] = []
    for c in range(CYCLES):
        for f in cycle:
            g = dict(f)
            original = int(g.get("frame", 0))
            g["original_frame"] = original
            g["cycle_index"] = c
            g["virtual_frame"] = len(timeline)
            timeline.append(g)

    OUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    value_counts = Counter()
    rows = 0
    with OUT_DATASET.open("w", encoding="utf-8") as out:
        for i in range(0, len(timeline) - SEQ_LEN):
            seq = timeline[i:i + SEQ_LEN]
            target = timeline[i + SEQ_LEN]
            action = action_from_frame(target)
            value_counts.update(action)
            row = {
                "version": 44,
                "seq_len": SEQ_LEN,
                "cycle_start": CYCLE_START,
                "cycle_end": CYCLE_END,
                "states": [state_from_frame(f) for f in seq],
                "action": action,
                "target_original_frame": int(target.get("original_frame", target.get("frame", 0))),
                "target_virtual_frame": int(target.get("virtual_frame", 0)),
                "cycle_index": int(target.get("cycle_index", 0)),
            }
            out.write(json.dumps(row, separators=(",", ":")) + "\n")
            rows += 1

    summary = {
        "version": 44,
        "reference": str(REF_PATH),
        "dataset": str(OUT_DATASET),
        "rows": rows,
        "seq_len": SEQ_LEN,
        "state_dim": len(state_from_frame(timeline[0])),
        "action_dim": 20,
        "cycle_start": CYCLE_START,
        "cycle_end": CYCLE_END,
        "cycle_source_frames": len(cycle),
        "cycles": CYCLES,
        "timeline_frames": len(timeline),
        "value_counts": value_counts.most_common(),
        "point_order": POINT_ORDER,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Dataset:", OUT_DATASET)
    print("Summary:", OUT_SUMMARY)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
