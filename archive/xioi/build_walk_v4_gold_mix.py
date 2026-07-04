#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"

GOLD = ROOT / "datasets/ml/walk_xioi_ff_until_427.jsonl"
MOTION = ROOT / "datasets/ml/walk_motion_v2.jsonl"

OUT = ROOT / "datasets/ml/walk_v4_gold_mix.jsonl"
SUMMARY = ROOT / "datasets/ml/walk_v4_gold_mix_summary.json"

GOLD_REPEAT = 25
MOTION_LIMIT = 3000


def read_lines(path):
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    gold = read_lines(GOLD)
    motion = read_lines(MOTION)[:MOTION_LIMIT]

    lines = []

    for _ in range(GOLD_REPEAT):
        lines.extend(gold)

    lines.extend(motion)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    summary = {
        "gold_examples": len(gold),
        "gold_repeat": GOLD_REPEAT,
        "gold_total": len(gold) * GOLD_REPEAT,
        "motion_examples": len(motion),
        "total": len(lines),
        "output": str(OUT),
    }

    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
