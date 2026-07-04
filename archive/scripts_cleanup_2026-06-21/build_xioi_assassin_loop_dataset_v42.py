#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

ROOT = Path.home() / "Documents" / "ToribashAI"
REF = ROOT / "generated_replays" / "xioi_assassin_reference_v42_0_315.json"
OUT = ROOT / "datasets" / "ml" / "xioi_assassin_loop_v42_sequences.jsonl"
SUMMARY = ROOT / "generated_replays" / "xioi_assassin_loop_v42_dataset_summary.json"
SEQ_LEN = 8
CYCLES = 5


def state_vec(frame: dict) -> list[float]:
    vals: list[float] = []
    for name in frame["point_order"] if "point_order" in frame else []:
        pass
    return vals


def main() -> None:
    data = json.loads(REF.read_text(encoding="utf-8"))
    point_order = data["point_order"]
    base = data["frames"]
    if len(base) < SEQ_LEN + 2:
        raise RuntimeError("Not enough reference frames")

    # Repeat the true walk segment several times so the GRU sees the end -> start transition.
    looped = []
    for cycle in range(CYCLES):
        for fr in base:
            item = dict(fr)
            item["cycle"] = cycle
            item["loop_frame"] = fr["frame"]
            looped.append(item)

    # Normalize positions relative to first chest position of each local sequence in builder below.
    rows = []
    counts = Counter()
    for i in range(0, len(looped) - SEQ_LEN):
        seq_frames = looped[i:i+SEQ_LEN]
        target = looped[i+SEQ_LEN]
        origin = seq_frames[0]["points"].get("chest", [0.0, 0.0, 0.0])
        seq = []
        for fr in seq_frames:
            st = []
            for name in point_order:
                p = fr["points"].get(name, [0.0, 0.0, 0.0])
                st.extend([p[0]-origin[0], p[1]-origin[1], p[2]-origin[2]])
            st.extend(fr.get("prev_action", [0]*20))
            seq.append(st)
        action = target.get("action", [0]*20)
        for v in action:
            counts[int(v)] += 1
        rows.append({
            "source": str(REF),
            "seq_len": SEQ_LEN,
            "state_dim": len(seq[0]),
            "target_frame": target["frame"],
            "target_cycle": target.get("cycle", 0),
            "sequence": seq,
            "action": action,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    summary = {
        "version": 42,
        "source": str(REF),
        "dataset": str(OUT),
        "rows": len(rows),
        "seq_len": SEQ_LEN,
        "cycles": CYCLES,
        "state_dim": rows[0]["state_dim"] if rows else None,
        "action_dim": 20,
        "value_counts": counts.most_common(),
        "point_order": point_order,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Dataset:", OUT)
    print("Summary:", SUMMARY)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
