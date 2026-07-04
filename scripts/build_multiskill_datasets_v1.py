#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "datasets" / "skills"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REPLAY_ROOT = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"

JOINTS = list(range(20))
SEQ_LEN = 8

SKILLS = {
    "launch": {
        "source": REPLAY_ROOT / "parkour" / "xioi_427_assassincreedhunter_v37.rpl",
        "start": 0,
        "end": 315,
    },
    "walk": {
        "source": REPLAY_ROOT / "parkour" / "xioi_loop_len265_champion_v51.rpl",
        "start": 0,
        "end": 265,
    },
}


def parse_rpl(path: Path):
    frames = {}
    current_frame = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            if line.startswith("FRAME "):
                try:
                    current_frame = int(line.split()[1].replace(";", ""))
                    frames.setdefault(current_frame, {})
                except Exception:
                    current_frame = None

            elif line.startswith("JOINT ") and current_frame is not None:
                try:
                    after = line.split(";", 1)[1].strip()
                    parts = after.split()

                    pairs = []
                    for i in range(0, len(parts) - 1, 2):
                        joint = int(parts[i])
                        value = int(parts[i + 1])
                        if joint in JOINTS:
                            pairs.append((joint, value))

                    if pairs:
                        frames[current_frame].update(dict(pairs))
                except Exception:
                    pass

    if not frames:
        return []

    ordered = []
    last = {j: 3 for j in JOINTS}

    for frame_id in sorted(frames.keys()):
        current = last.copy()
        current.update(frames[frame_id])
        ordered.append({
            "frame": frame_id,
            "joints": [current[j] for j in JOINTS],
        })
        last = current

    return ordered


def build_skill(skill_name, cfg):
    source = cfg["source"]

    if not source.exists():
        print(f"[ERROR] Source introuvable pour {skill_name}: {source}")
        return

    frames = parse_rpl(source)
    if not frames:
        print(f"[WARN] Replay illisible: {source}")
        return

    start = cfg["start"]
    end = cfg["end"]

    sliced = [f for f in frames if start <= f["frame"] <= end]

    if len(sliced) <= SEQ_LEN:
        print(f"[WARN] Pas assez de frames pour {skill_name}: {len(sliced)}")
        return

    out_path = OUT_DIR / f"{skill_name}_skill_v1.jsonl"
    summary_path = OUT_DIR / f"{skill_name}_skill_v1_summary.json"

    rows = 0
    active_counter = Counter()

    with out_path.open("w", encoding="utf-8") as out:
        for i in range(len(sliced) - SEQ_LEN):
            seq = sliced[i:i + SEQ_LEN]
            target = sliced[i + SEQ_LEN]

            action = target["joints"]
            active_count = sum(1 for v in action if v != 3)
            active_counter[active_count] += 1

            row = {
                "skill": skill_name,
                "source": str(source),
                "seq_len": SEQ_LEN,
                "input_frames": [x["frame"] for x in seq],
                "target_frame": target["frame"],
                "state": [x["joints"] for x in seq],
                "action": action,
                "active_count": active_count,
            }

            out.write(json.dumps(row) + "\n")
            rows += 1

    summary = {
        "skill": skill_name,
        "source": str(source),
        "start": start,
        "end": end,
        "seq_len": SEQ_LEN,
        "frames_used": len(sliced),
        "first_frame": sliced[0]["frame"],
        "last_frame": sliced[-1]["frame"],
        "rows": rows,
        "active_count_distribution": dict(sorted(active_counter.items())),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] {skill_name}")
    print(f"     source: {source}")
    print(f"     frames: {len(sliced)}")
    print(f"     rows:   {rows}")
    print(f"     out:    {out_path}")


def main():
    print("=== ToribashAI multiskill dataset builder V1 ===")
    print(f"Replay root: {REPLAY_ROOT}")
    print(f"Output:      {OUT_DIR}")

    for skill_name, cfg in SKILLS.items():
        build_skill(skill_name, cfg)

    print("\nTerminé 💜")


if __name__ == "__main__":
    main()
