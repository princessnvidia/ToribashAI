#!/usr/bin/env python3
from pathlib import Path
import json

PROJECT = Path.home() / "Documents" / "ToribashAI"

INPUT = PROJECT / "datasets" / "locomotion" / "ground_walk_dataset_v1.jsonl"
OUT = PROJECT / "datasets" / "locomotion" / "run_patterns_v1.jsonl"
SUMMARY = PROJECT / "datasets" / "locomotion" / "run_patterns_v1_summary.json"

def main():
    rows = []
    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["label"] == "run":
                rows.append(row)

    rows.sort(key=lambda r: r["forward_speed"], reverse=True)

    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(INPUT),
        "output": str(OUT),
        "run_patterns": len(rows),
        "top_20": [
            {
                "source_name": r["source_name"],
                "start_frame": r["start_frame"],
                "end_frame": r["end_frame"],
                "forward_speed": r["forward_speed"],
                "z_min": r["features"]["z_min"],
                "z_range": r["features"]["z_range"],
                "leg_activity": r["features"]["leg_activity"],
                "support_change_rate": r["features"]["support_change_rate"],
                "forward_lean": r["features"]["forward_lean"],
            }
            for r in rows[:20]
        ],
    }

    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Run patterns extraits.")
    print(f"Count: {len(rows)}")
    print(f"OUT: {OUT}")
    print(f"SUMMARY: {SUMMARY}")

if __name__ == "__main__":
    main()
