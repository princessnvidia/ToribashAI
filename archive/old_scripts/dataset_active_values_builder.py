#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

PROJECT_DIR = Path.home() / "Documents" / "ToribashAI"

INPUT_PATH = PROJECT_DIR / "datasets" / "ml" / "parkour_sequences_len8.jsonl"

OUTPUT_PATH = PROJECT_DIR / "datasets" / "ml" / "parkour_active_values_len8.jsonl"

SUMMARY_PATH = PROJECT_DIR / "datasets" / "ml" / "parkour_active_values_len8_summary.json"


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_sequences = 0
    total_examples = 0

    joint_counter = Counter()
    value_counter = Counter()

    with INPUT_PATH.open("r", encoding="utf-8") as fin, \
         OUTPUT_PATH.open("w", encoding="utf-8") as fout:

        for line in fin:
            if not line.strip():
                continue

            obj = json.loads(line)

            states = obj.get("states")
            if states is None:
                states = obj.get("state_seq")

            action = obj.get("action")

            if states is None or action is None:
                continue

            total_sequences += 1

            for joint_id, value in enumerate(action):

                value = int(value)

                if value == 0:
                    continue

                example = {
                    "states": states,
                    "joint_id": joint_id,
                    "target_value": value,
                    "target_class": value - 1
                }

                fout.write(
                    json.dumps(example, ensure_ascii=False)
                    + "\n"
                )

                total_examples += 1

                joint_counter[joint_id] += 1
                value_counter[value] += 1

    summary = {
        "input": str(INPUT_PATH),
        "output": str(OUTPUT_PATH),
        "total_sequences": total_sequences,
        "total_examples": total_examples,
        "joint_distribution": sorted(
            joint_counter.items(),
            key=lambda x: x[1],
            reverse=True
        ),
        "value_distribution": sorted(
            value_counter.items(),
            key=lambda x: x[1],
            reverse=True
        )
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print("Done.")
    print(f"Sequences: {total_sequences}")
    print(f"Examples actifs: {total_examples}")
    print(f"Dataset: {OUTPUT_PATH}")
    print(f"Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
