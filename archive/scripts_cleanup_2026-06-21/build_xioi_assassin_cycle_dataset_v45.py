#!/usr/bin/env python3
"""
build_xioi_assassin_cycle_dataset_v45.py

V45 fixes V44: build the cycle dataset from REAL JOINT actions in the RPL,
not from the reference-position JSON. V44 accidentally produced only zeros.

Source:
  generated_replays/xioi_427_assassincreedhunter_v37.rpl
Cycle:
  frames 70 -> 295, repeated several times.
Output:
  datasets/ml/xioi_assassin_cycle_v45_sequences.jsonl
  generated_replays/xioi_assassin_cycle_v45_dataset_summary.json
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
SRC_RPL = ROOT / "generated_replays" / "xioi_427_assassincreedhunter_v37.rpl"
OUT_DATASET = ROOT / "datasets" / "ml" / "xioi_assassin_cycle_v45_sequences.jsonl"
OUT_SUMMARY = ROOT / "generated_replays" / "xioi_assassin_cycle_v45_dataset_summary.json"

CYCLE_START = 70
CYCLE_END = 295
CYCLES = 10
SEQ_LEN = 8
ACTION_DIM = 20

# Same point order idea as previous datasets, but V45 trains action rhythm.
# State = last SEQ_LEN action vectors normalized to 0..4. This avoids fake POS loops.
STATE_DIM = ACTION_DIM


def parse_rpl_actions(path: Path) -> dict[int, list[int]]:
    if not path.exists():
        raise FileNotFoundError(path)
    actions: dict[int, list[int]] = {}
    current_frame: int | None = None

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if line.startswith("FRAME "):
            try:
                current_frame = int(line.split()[1].split(";")[0])
                actions.setdefault(current_frame, [0] * ACTION_DIM)
            except Exception:
                current_frame = None
            continue

        if current_frame is None:
            continue

        if line.startswith("JOINT 0;"):
            vals = line.split(";", 1)[1].strip().split()
            nums = []
            for x in vals:
                try:
                    nums.append(int(float(x)))
                except Exception:
                    pass
            # Pairs: joint value joint value ...
            if len(nums) >= 2:
                a = actions.setdefault(current_frame, [0] * ACTION_DIM)
                for i in range(0, len(nums) - 1, 2):
                    j, v = nums[i], nums[i + 1]
                    if 0 <= j < ACTION_DIM and 0 <= v <= 4:
                        a[j] = v
    return actions


def build_cycle(actions_by_frame: dict[int, list[int]]) -> list[dict]:
    source_frames = [f for f in sorted(actions_by_frame) if CYCLE_START <= f <= CYCLE_END]
    if not source_frames:
        raise RuntimeError(f"No source frames in cycle {CYCLE_START}-{CYCLE_END}")

    timeline: list[dict] = []
    out_frame = 0
    for cycle in range(CYCLES):
        for f in source_frames:
            action = actions_by_frame[f]
            timeline.append({
                "frame": out_frame,
                "source_frame": f,
                "cycle": cycle,
                "action": action,
            })
            out_frame += 5
    return timeline


def make_rows(timeline: list[dict]) -> list[dict]:
    rows = []
    for i in range(SEQ_LEN, len(timeline)):
        seq = [timeline[k]["action"] for k in range(i - SEQ_LEN, i)]
        target = timeline[i]["action"]
        rows.append({
            "seq": seq,
            "target": target,
            "frame": timeline[i]["frame"],
            "source_frame": timeline[i]["source_frame"],
            "cycle": timeline[i]["cycle"],
        })
    return rows


def main() -> None:
    actions = parse_rpl_actions(SRC_RPL)
    timeline = build_cycle(actions)
    rows = make_rows(timeline)

    OUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DATASET.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    counts = Counter()
    active_counts = Counter()
    for row in rows:
        for v in row["target"]:
            counts[v] += 1
        active_counts[sum(1 for v in row["target"] if v != 0)] += 1

    summary = {
        "version": 45,
        "source_rpl": str(SRC_RPL),
        "dataset": str(OUT_DATASET),
        "rows": len(rows),
        "seq_len": SEQ_LEN,
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "cycle_start": CYCLE_START,
        "cycle_end": CYCLE_END,
        "cycle_source_frames": len([f for f in sorted(actions) if CYCLE_START <= f <= CYCLE_END]),
        "cycles": CYCLES,
        "value_counts": counts.most_common(),
        "active_count_distribution": active_counts.most_common(),
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Dataset:", OUT_DATASET)
    print("Summary:", OUT_SUMMARY)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
