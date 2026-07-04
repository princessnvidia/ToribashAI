#!/usr/bin/env python3
from pathlib import Path
import json
from collections import Counter, defaultdict

PROJECT = Path.home() / "Documents" / "ToribashAI"

INPUT = PROJECT / "datasets" / "locomotion" / "run_legs_dataset_v1.jsonl"
OUT = PROJECT / "datasets" / "locomotion" / "run_legs_sequences_v1.jsonl"
SUMMARY = PROJECT / "datasets" / "locomotion" / "run_legs_sequences_v1_summary.json"

SEQ_LEN = 8


def feature_vec(row):
    f = row["features"]
    return (
        [int(v) for v in row["leg_now"]]
        + [
            float(f["forward_speed"]),
            float(f["leg_activity"]),
            float(f["support_change_rate"]),
            float(f["forward_lean"]),
            float(f["z_min"]),
            float(f["z_range"]),
        ]
    )


def main():
    groups = defaultdict(list)

    with INPUT.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["source_name"], row["start_frame"], row["end_frame"])
            groups[key].append(row)

    rows_out = []
    target_counts = Counter()

    for key, rows in groups.items():
        rows.sort(key=lambda r: int(r["t"]))

        if len(rows) < SEQ_LEN:
            continue

        for i in range(0, len(rows) - SEQ_LEN + 1):
            chunk = rows[i:i + SEQ_LEN]

            x_seq = [feature_vec(r) for r in chunk]
            y = [int(v) for v in chunk[-1]["leg_next"]]

            for v in y:
                target_counts[str(v)] += 1

            rows_out.append({
                "source_name": key[0],
                "start_frame": key[1],
                "end_frame": key[2],
                "t_start": int(chunk[0]["t"]),
                "t_end": int(chunk[-1]["t"]),
                "seq_len": SEQ_LEN,
                "x_seq": x_seq,
                "leg_next": y,
            })

    with OUT.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input": str(INPUT),
        "output": str(OUT),
        "seq_len": SEQ_LEN,
        "groups": len(groups),
        "samples": len(rows_out),
        "target_counts": dict(target_counts),
    }

    SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Run legs sequences créées.")
    print(f"Samples: {len(rows_out)}")
    print(f"OUT: {OUT}")
    print(f"SUMMARY: {SUMMARY}")
    print("Target counts:", dict(target_counts))


if __name__ == "__main__":
    main()
