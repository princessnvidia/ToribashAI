#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT = Path.home() / "Documents/ToribashAI"
GEN = ROOT / "generated_replays"
DATA = ROOT / "datasets/ml"
REF = GEN / "xioi_loop_len265_champion_v51_reference.json"
OUT = DATA / "xioi_loop_len265_v52_sequences.jsonl"
SUMMARY = GEN / "xioi_loop_len265_v52_dataset_summary.json"
SEQ_LEN = 8


def main() -> None:
    if not REF.exists():
        raise FileNotFoundError(f"Missing reference: {REF}\nRun generate_xioi_loop_len265_champion_v51.py first.")
    DATA.mkdir(parents=True, exist_ok=True)
    ref = json.loads(REF.read_text(encoding="utf-8"))
    frames = sorted(ref.get("frames", []), key=lambda x: x["frame"])
    if len(frames) <= SEQ_LEN:
        raise RuntimeError("Not enough frames for sequence dataset")

    rows = []
    counts = Counter()
    active_counts = Counter()
    for i in range(SEQ_LEN, len(frames)):
        seq = [frames[k]["values"] for k in range(i - SEQ_LEN, i)]
        target = frames[i]["values"]
        row = {
            "source": str(REF),
            "frame": frames[i]["frame"],
            "seq": seq,
            "action": target,
            "pairs": frames[i].get("pairs", []),
        }
        rows.append(row)
        for v in target:
            counts[int(v)] += 1
        active_counts[sum(1 for v in target if v != 0)] += 1

    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "version": 52,
        "reference": str(REF),
        "dataset": str(OUT),
        "rows": len(rows),
        "seq_len": SEQ_LEN,
        "state_dim": 20,
        "action_dim": 20,
        "frame_min": frames[0]["frame"],
        "frame_max": frames[-1]["frame"],
        "value_counts": counts.most_common(),
        "active_count_distribution": active_counts.most_common(),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Dataset:", OUT)
    print("Summary:", SUMMARY)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
