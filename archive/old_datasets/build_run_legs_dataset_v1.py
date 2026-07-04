#!/usr/bin/env python3
from pathlib import Path
import json
from collections import Counter

PROJECT = Path.home() / "Documents" / "ToribashAI"

INPUT = PROJECT / "datasets" / "locomotion" / "run_patterns_v1.jsonl"
OUT = PROJECT / "datasets" / "locomotion" / "run_legs_dataset_v1.jsonl"
SUMMARY = PROJECT / "datasets" / "locomotion" / "run_legs_dataset_v1_summary.json"

LEG_JOINTS = [14, 15, 16, 17, 18, 19]


def main():
    rows_out = []
    action_counts = Counter()

    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            actions_seq = row["actions"]

            for t in range(len(actions_seq) - 1):
                current = actions_seq[t]
                nxt = actions_seq[t + 1]

                leg_now = [int(current[j]) for j in LEG_JOINTS]
                leg_next = [int(nxt[j]) for j in LEG_JOINTS]

                for v in leg_next:
                    action_counts[str(v)] += 1

                rows_out.append({
                    "source_name": row["source_name"],
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "t": t,
                    "features": {
                        "forward_speed": row["forward_speed"],
                        "leg_activity": row["features"]["leg_activity"],
                        "support_change_rate": row["features"]["support_change_rate"],
                        "forward_lean": row["features"]["forward_lean"],
                        "z_min": row["features"]["z_min"],
                        "z_range": row["features"]["z_range"],
                    },
                    "leg_now": leg_now,
                    "leg_next": leg_next,
                })

    with OUT.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(INPUT),
        "output": str(OUT),
        "samples": len(rows_out),
        "leg_joints": LEG_JOINTS,
        "action_counts_next": dict(action_counts),
    }

    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Run legs dataset créé.")
    print(f"Samples: {len(rows_out)}")
    print(f"OUT: {OUT}")
    print(f"SUMMARY: {SUMMARY}")
    print("Action counts:", dict(action_counts))


if __name__ == "__main__":
    main()
