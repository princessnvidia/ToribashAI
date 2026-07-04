#!/usr/bin/env python3
from pathlib import Path
import json
import re

ROOT = Path.home() / "Documents" / "ToribashAI"
SRC = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay/archive_xioi_history/parkour/xioi_stable_loop_v49_base.rpl"
OUT = ROOT / "datasets" / "skills" / "xioi_v49_motion_windows_v1.json"

def floats(s):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]

frames = []
current = None

for raw in SRC.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = raw.strip()

    if line.startswith("FRAME "):
        m = re.search(r"FRAME\s+(-?\d+)", line)
        if m:
            current = {"frame": int(m.group(1)), "pos": None, "linvel": None}
            frames.append(current)

    elif current is not None and line.startswith("POS "):
        vals = floats(line)
        current["pos"] = vals

    elif current is not None and line.startswith("LINVEL "):
        vals = floats(line)
        current["linvel"] = vals

valid = [f for f in frames if f["pos"] and len(f["pos"]) >= 3]

# POS contient beaucoup de valeurs. On regroupe par triplets xyz
# et on prend le centre moyen des premiers body parts Tori.
def center(pos):
    pts = []
    for k in range(0, min(len(pos), 60), 3):
        pts.append((pos[k], pos[k+1], pos[k+2]))
    if not pts:
        return (0.0, 0.0, 0.0)
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
        sum(p[2] for p in pts) / len(pts),
    )

windows = []
W = 64

for i in range(0, len(valid) - W, 8):
    a = valid[i]
    b = valid[i + W]

    ax, ay, az = center(a["pos"])
    bx, by, bz = center(b["pos"])

    dy = by - ay
    dz = bz - az
    speed = dy / max(1, b["frame"] - a["frame"])

    windows.append({
        "start_idx": i,
        "end_idx": i + W,
        "start_frame": a["frame"],
        "end_frame": b["frame"],
        "dy": dy,
        "dz": dz,
        "speed_y": speed,
        "start_z": az,
        "end_z": bz,
    })

windows.sort(key=lambda x: x["dy"], reverse=True)
OUT.write_text(json.dumps(windows[:80], indent=2), encoding="utf-8")

print(f"[OK] frames valid: {len(valid)}")
print(f"[OK] écrit: {OUT}")
print("Top 20 fenêtres par déplacement Y:")
for w in windows[:20]:
    print(
        f"{w['start_frame']}→{w['end_frame']} "
        f"dy={w['dy']:.3f} speed={w['speed_y']:.4f} "
        f"z={w['start_z']:.2f}->{w['end_z']:.2f}"
    )
