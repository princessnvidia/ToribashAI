#!/usr/bin/env python3
from pathlib import Path
import json
from collections import Counter

BASE = Path.home() / "Documents/ToribashAI"

IN_DATASET = BASE / "datasets/ml/parkour_sequences_len8.jsonl"
OUT_DATASET = BASE / "datasets/ml/parkour_active_joints_len8.jsonl"
SUMMARY = BASE / "datasets/ml/parkour_active_joints_len8_summary.json"

SEQ_LEN = 8
JOINTS = 20


def main():
    total = 0
    active_counts = Counter()
    active_joint_counts = Counter()
    action_value_counts = Counter()

    with IN_DATASET.open("r", encoding="utf-8") as fin, \
         OUT_DATASET.open("w", encoding="utf-8") as fout:

        for line in fin:
            row = json.loads(line)

            action = row["action"]

            active = [
                0 if value == 0 else 1
                for value in action
            ]

            active_count = sum(active)

            out_row = {
                "source_json": row["source_json"],
                "source_rpl": row["source_rpl"],
                "fightname": row["fightname"],
                "mod": row["mod"],
                "start_frame": row["start_frame"],
                "end_frame": row["end_frame"],
                "seq_len": row["seq_len"],
                "states": row["states"],
                "action": action,
                "active_joints": active,
                "active_count": active_count,
            }

            fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            total += 1
            active_counts[active_count] += 1

            for joint_id, is_active in enumerate(active):
                if is_active:
                    active_joint_counts[joint_id] += 1

            for value in action:
                action_value_counts[value] += 1

    summary = {
        "input": str(IN_DATASET),
        "output": str(OUT_DATASET),
        "seq_len": SEQ_LEN,
        "joints": JOINTS,
        "total_sequences": total,
        "active_count_distribution": active_counts.most_common(),
        "active_joint_counts": active_joint_counts.most_common(),
        "action_value_counts": action_value_counts.most_common(),
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Dataset active joints:", OUT_DATASET)
    print("Résumé:", SUMMARY)
    print("Séquences:", total)
    print("Distribution active_count:", active_counts.most_common())


if __name__ == "__main__":
    main()
