#!/usr/bin/env python3
from pathlib import Path
import json
import re
import math

ROOT = Path.home() / "Documents" / "ToribashAI"
SRC = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay/archive_xioi_history/parkour/xioi_stable_loop_v49_base.rpl"
OUT = ROOT / "datasets" / "skills" / "walk_loop_candidates_v1.json"

def nums(s):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]

frames = []
current = None

for raw in SRC.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = raw.strip()

    if line.startswith("FRAME "):
        n = nums(line)
        if n:
            current = {"frame": int(n[0]), "pos": None, "joints": None}
            frames.append(current)

    elif current is not None and line.startswith("POS "):
        vals = nums(line)
        if vals and vals[0] in (0, 1):
            vals = vals[1:]
        current["pos"] = vals[:63]

    elif current is not None and line.startswith("JOINT "):
        parts = line.split(";", 1)[1].strip().split()
        joints = [3] * 20
        for i in range(0, len(parts) - 1, 2):
            j = int(parts[i])
            v = int(parts[i + 1])
            if 0 <= j < 20:
                joints[j] = v
        current["joints"] = joints

frames = [f for f in frames if f["pos"] and f["joints"]]

def center(pos):
    pts = []
    for k in range(0, 63, 3):
        pts.append((pos[k], pos[k + 1], pos[k + 2]))
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
        sum(p[2] for p in pts) / len(pts),
    )

def joint_distance(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)

WINDOW = 64
STEP = 8

candidates = []

for i in range(0, len(frames) - WINDOW * 2, STEP):
    a0 = frames[i]
    a1 = frames[i + WINDOW]
    b1 = frames[i + WINDOW * 2]

    ax, ay, az = center(a0["pos"])
    bx, by, bz = center(a1["pos"])
    cx, cy, cz = center(b1["pos"])

    dy1 = by - ay
    dy2 = cy - by
    dz1 = bz - az
    dz2 = cz - bz

    speed_stability = abs(dy1 - dy2)
    height_stability = abs(dz1) + abs(dz2) + abs(bz - az)

    jd = joint_distance(a0["joints"], a1["joints"])

    score = 0
    score += dy1 * 3.0
    score += dy2 * 3.0
    score -= speed_stability * 10.0
    score -= height_stability * 2.0
    score -= jd * 0.5

    candidates.append({
        "start_frame": a0["frame"],
        "mid_frame": a1["frame"],
        "end_frame": b1["frame"],
        "dy1": dy1,
        "dy2": dy2,
        "speed_stability": speed_stability,
        "z0": az,
        "z1": bz,
        "z2": cz,
        "height_stability": height_stability,
        "joint_distance_start_mid": jd,
        "score": score,
    })

candidates.sort(key=lambda x: x["score"], reverse=True)

OUT.write_text(json.dumps(candidates[:80], indent=2), encoding="utf-8")

print(f"[OK] frames: {len(frames)}")
print(f"[OK] out: {OUT}")
print("Top 20 loop candidates:")
for c in candidates[:20]:
    print(
        f"{c['start_frame']}→{c['end_frame']} "
        f"score={c['score']:.2f} "
        f"dy={c['dy1']:.2f}/{c['dy2']:.2f} "
        f"z={c['z0']:.2f}->{c['z1']:.2f}->{c['z2']:.2f} "
        f"jd={c['joint_distance_start_mid']}"
    )
