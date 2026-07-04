#!/usr/bin/env python3
from pathlib import Path
import json

PROJECT = Path.home() / "Documents" / "ToribashAI"

INPUT_JSONL = (
    PROJECT
    / "datasets"
    / "motion_patterns"
    / "ground_walk_patterns_v1.jsonl"
)

OUT_DIR = PROJECT / "datasets" / "locomotion"

OUT_JSONL = OUT_DIR / "ground_walk_dataset_v1.jsonl"
OUT_SUMMARY = OUT_DIR / "ground_walk_dataset_v1_summary.json"


def label_from_speed(forward):
    if forward < 3.0:
        return "idle"

    if forward < 8.0:
        return "walk"

    return "run"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    counts = {}

    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            item = json.loads(line)

            feat = item["features"]

            forward = float(feat["forward_y_negative"])

            label = label_from_speed(forward)

            counts[label] = counts.get(label, 0) + 1

            rows.append({
                "source_name": item["source_name"],
                "start_frame": item["start_frame"],
                "end_frame": item["end_frame"],
                "label": label,
                "forward_speed": forward,
                "actions": item["actions"],
                "centers": item["centers"],
                "features": {
                    "leg_activity": feat["leg_activity"],
                    "core_activity": feat["core_activity"],
                    "support_change_rate": feat["support_change_rate"],
                    "forward_lean": feat["forward_lean"],
                    "z_min": feat["z_min"],
                    "z_range": feat["z_range"],
                }
            })

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "samples": len(rows),
        "labels": counts,
        "output": str(OUT_JSONL),
    }

    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Dataset locomotion créé.")
    print(f"Samples: {len(rows)}")
    print(f"Labels: {counts}")
    print(f"JSONL: {OUT_JSONL}")


if __name__ == "__main__":
    main()
