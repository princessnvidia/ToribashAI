#!/usr/bin/env python3
from pathlib import Path
import json
import re
from collections import Counter

ROOT = Path.home() / "Documents" / "ToribashAI"
SRC = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay/archive_xioi_history/parkour/xioi_stable_loop_v49_base.rpl"
OUT_DIR = ROOT / "datasets" / "skills"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 8
JOINTS = list(range(20))

SKILLS = {
    "launch": [(0, 320)],
    "walk": [
        (520, 840),
        (1560, 1880),
        (3120, 3440),
        (3920, 4240),
        (4440, 4760),
        (6000, 6320),
        (6800, 7120),
        (7840, 8160),
    ],
}

def nums(s):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]

frames = []
current = None
last_joints = {j: 3 for j in JOINTS}

for raw in SRC.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = raw.strip()

    if line.startswith("FRAME "):
        n = nums(line)
        if n:
            current = {
                "frame": int(n[0]),
                "joints": last_joints.copy(),
                "pos": None,
                "qat": None,
                "linvel": None,
                "angvel": None,
            }
            frames.append(current)

    elif current is not None and line.startswith("JOINT "):
        parts = line.split(";", 1)[1].strip().split()
        for i in range(0, len(parts) - 1, 2):
            j = int(parts[i])
            v = int(parts[i + 1])
            if j in last_joints:
                last_joints[j] = v
                current["joints"][j] = v

    elif current is not None and line.startswith("POS "):
        vals = nums(line)
        if vals and vals[0] in (0, 1):
            vals = vals[1:]
        current["pos"] = vals[:63]

    elif current is not None and line.startswith("QAT "):
        vals = nums(line)
        if vals and vals[0] in (0, 1):
            vals = vals[1:]
        current["qat"] = vals[:84]

    elif current is not None and line.startswith("LINVEL "):
        vals = nums(line)
        if vals and vals[0] in (0, 1):
            vals = vals[1:]
        current["linvel"] = vals[:63]

    elif current is not None and line.startswith("ANGVEL "):
        vals = nums(line)
        if vals and vals[0] in (0, 1):
            vals = vals[1:]
        current["angvel"] = vals[:63]

frames = [f for f in frames if f["pos"] and f["qat"] and f["linvel"] and f["angvel"]]

def frame_state(f):
    return {
        "frame": f["frame"],
        "joints": [f["joints"][j] for j in JOINTS],
        "pos": f["pos"],
        "qat": f["qat"],
        "linvel": f["linvel"],
        "angvel": f["angvel"],
    }

def build(skill, windows):
    rows = []
    active_counter = Counter()

    for start, end in windows:
        chunk = [f for f in frames if start <= f["frame"] <= end]

        for i in range(len(chunk) - SEQ_LEN):
            seq = chunk[i:i + SEQ_LEN]
            target = chunk[i + SEQ_LEN]

            action = [target["joints"][j] for j in JOINTS]
            active = sum(1 for v in action if v != 3)
            active_counter[active] += 1

            rows.append({
                "skill": skill,
                "source": str(SRC),
                "window": [start, end],
                "seq_len": SEQ_LEN,
                "input_frames": [x["frame"] for x in seq],
                "target_frame": target["frame"],
                "state": [frame_state(x) for x in seq],
                "action": action,
                "active_count": active,
            })

    out = OUT_DIR / f"{skill}_skill_v2.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    summary = {
        "skill": skill,
        "source": str(SRC),
        "windows": windows,
        "seq_len": SEQ_LEN,
        "rows": len(rows),
        "active_count_distribution": dict(sorted(active_counter.items())),
    }

    summary_path = OUT_DIR / f"{skill}_skill_v2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[OK] {skill}: rows={len(rows)} -> {out}")

def main():
    print("=== Build Xioi Skills V2 ===")
    print(f"frames parsed with physics: {len(frames)}")
    build("launch", SKILLS["launch"])
    build("walk", SKILLS["walk"])

if __name__ == "__main__":
    main()
