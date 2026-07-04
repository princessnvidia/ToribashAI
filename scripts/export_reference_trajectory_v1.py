#!/usr/bin/env python3
from pathlib import Path
import json
import re

ROOT = Path.home() / "Documents" / "ToribashAI"
SRC = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay/archive_xioi_history/parkour/xioi_stable_loop_v49_base.rpl"
OUT = ROOT / "evolution" / "reference_trajectory_v1.json"

SEGMENTS = {
    "launch_transition": (0, 1680),
    "walk_loop": (1680, 2320),
}

def nums(s):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]

def center_from_pos(pos):
    pts = []
    for k in range(0, 63, 3):
        pts.append((pos[k], pos[k + 1], pos[k + 2]))
    return {
        "x": sum(p[0] for p in pts) / len(pts),
        "y": sum(p[1] for p in pts) / len(pts),
        "z": sum(p[2] for p in pts) / len(pts),
    }

frames = []
current = None

for raw in SRC.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = raw.strip()

    if line.startswith("FRAME "):
        n = nums(line)
        if n:
            current = {"frame": int(n[0]), "pos": None}
            frames.append(current)

    elif current is not None and line.startswith("POS "):
        vals = nums(line)
        if vals and vals[0] in (0, 1):
            vals = vals[1:]
        current["pos"] = vals[:63]

frames = [f for f in frames if f["pos"] and len(f["pos"]) >= 63]

out = {
    "source": str(SRC),
    "segments": {},
}

for name, (start, end) in SEGMENTS.items():
    chunk = [f for f in frames if start <= f["frame"] <= end]

    points = []
    first_center = None

    for f in chunk:
        c = center_from_pos(f["pos"])

        if first_center is None:
            first_center = c

        points.append({
            "frame": f["frame"] - start,
            "source_frame": f["frame"],
            "x": c["x"] - first_center["x"],
            "y": c["y"] - first_center["y"],
            "z": c["z"],
        })

    out["segments"][name] = {
        "start": start,
        "end": end,
        "points": points,
        "frames": len(points),
    }

OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

print(f"[OK] écrit: {OUT}")
for name, seg in out["segments"].items():
    print(name, "points=", seg["frames"])
    print(" first=", seg["points"][0])
    print(" last =", seg["points"][-1])
