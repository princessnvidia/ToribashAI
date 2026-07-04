#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"

SOURCE_PATH = ROOT / "parsed/#Xioi#Pk - YpSkA.json"
PARKOUR_DIR = ROOT / "datasets/parkour_json"

OUT_PATH = ROOT / "datasets/ml/walk_xioi_ff_until_427.jsonl"
SUMMARY_PATH = ROOT / "datasets/ml/walk_xioi_ff_until_427_summary.json"

SEQ_LEN = 8
MAX_FRAME = 427
PLAYER_ID = "0"


def find_source():
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(SOURCE_PATH)
    print("Source:", SOURCE_PATH)
    return SOURCE_PATH


def flatten_state(player):
    state = []

    for key in ["pos", "qat", "linvel", "angvel"]:
        arr = player.get(key, [])
        for item in arr:
            state.extend(float(x) for x in item)

    return state


def get_action(player):
    joints = player.get("joints", {})
    return [int(joints.get(str(i), 0)) for i in range(20)]


def main():
    src = find_source()
    data = json.loads(src.read_text(encoding="utf-8"))
    frames = data["frames"]

    frame_ids = sorted(
        [int(k) for k in frames.keys() if int(k) <= MAX_FRAME]
    )

    examples = []

    for idx in range(0, len(frame_ids) - SEQ_LEN):
        seq_frame_ids = frame_ids[idx:idx + SEQ_LEN]
        target_frame = frame_ids[idx + SEQ_LEN]

        states = []

        ok = True
        for fid in seq_frame_ids:
            fr = frames[str(fid)]
            player = fr.get("players", {}).get(PLAYER_ID)
            if not player:
                ok = False
                break
            states.append(flatten_state(player))

        if not ok:
            continue

        target_player = frames[str(target_frame)].get("players", {}).get(PLAYER_ID)
        if not target_player:
            continue

        action = get_action(target_player)

        ex = {
            "source_json": str(src),
            "source_rpl": data.get("metadata", {}).get("source_rpl", ""),
            "fightname": data.get("metadata", {}).get("fightname", src.stem),
            "mod": data.get("metadata", {}).get("mod", ""),
            "start_frame": seq_frame_ids[0],
            "end_frame": target_frame,
            "seq_len": SEQ_LEN,
            "states": states,
            "action": action,
            "gold_walk": True,
            "gold_source": "xioi_ff_until_427",
        }

        examples.append(ex)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    summary = {
        "source": str(src),
        "output": str(OUT_PATH),
        "max_frame": MAX_FRAME,
        "seq_len": SEQ_LEN,
        "examples": len(examples),
        "first_frame": frame_ids[0] if frame_ids else None,
        "last_frame": frame_ids[-1] if frame_ids else None,
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
