#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

PROJECT = Path.home() / "Documents" / "ToribashAI"

INPUT_DIR = PROJECT / "datasets" / "parkour_json"
OUT_DIR = PROJECT / "datasets" / "ml"
OUT_PATH = OUT_DIR / "walk_fullbody_sequences_v1.jsonl"
SUMMARY_PATH = OUT_DIR / "walk_fullbody_sequences_v1_summary.json"

SEQ_LEN = 8

CONTROL_JOINTS = [
    4, 5, 6, 7,      # bras
    14, 15, 16,      # jambe droite
    17, 18, 19,      # jambe gauche
]

DEFAULT_ACTION = 3


def sorted_frame_items(frames):
    if isinstance(frames, dict):
        return sorted(
            frames.items(),
            key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 0,
        )

    if isinstance(frames, list):
        return list(enumerate(frames))

    return []


def extract_player0_joints(frame):
    players = frame.get("players", {})
    player0 = players.get("0") or players.get(0)

    if not player0:
        return {}

    joints = player0.get("joints", {})

    out = {}
    for k, v in joints.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            pass

    return out


def action_vector(joints):
    return [
        int(joints.get(jid, DEFAULT_ACTION))
        for jid in CONTROL_JOINTS
    ]


def process_replay(path):
    data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    frames = data.get("frames", {})

    actions = []

    for _, frame in sorted_frame_items(frames):
        if not isinstance(frame, dict):
            continue

        joints = extract_player0_joints(frame)
        if not joints:
            continue

        actions.append(action_vector(joints))

    if len(actions) <= SEQ_LEN:
        return []

    rows = []

    for i in range(SEQ_LEN, len(actions)):
        rows.append({
            "replay": path.name,
            "seq_len": SEQ_LEN,
            "control_joints": CONTROL_JOINTS,
            "input": actions[i - SEQ_LEN:i],
            "target": actions[i],
        })

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    replay_paths = sorted(INPUT_DIR.glob("*.json"))

    total_replays = 0
    used_replays = 0
    total_rows = 0
    errors = 0
    value_counts = Counter()

    with OUT_PATH.open("w", encoding="utf-8") as out:
        for path in replay_paths:
            total_replays += 1

            try:
                rows = process_replay(path)
            except Exception as e:
                errors += 1
                print("ERROR:", path.name, e)
                continue

            if rows:
                used_replays += 1

            for row in rows:
                for v in row["target"]:
                    value_counts[int(v)] += 1

                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += 1

    summary = {
        "input_dir": str(INPUT_DIR),
        "output_path": str(OUT_PATH),
        "seq_len": SEQ_LEN,
        "control_joints": CONTROL_JOINTS,
        "total_replays": total_replays,
        "used_replays": used_replays,
        "total_sequences": total_rows,
        "errors": errors,
        "target_value_counts": dict(sorted(value_counts.items())),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("Dataset écrit:", OUT_PATH)
    print("Résumé:", SUMMARY_PATH)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
