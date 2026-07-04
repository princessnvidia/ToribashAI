#!/usr/bin/env python3
from pathlib import Path
import json
import re

ROOT = Path.home() / "Documents" / "ToribashAI"
SRC = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay/archive_xioi_history/parkour/xioi_stable_loop_v49_base.rpl"
OUT = ROOT / "datasets" / "skills" / "xioi_v49_motion_windows_v2.json"

def nums(s):
    return [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", s)]

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

        # POS commence souvent par "POS 0;" puis triplets x y z.
        # On enlève l'identifiant joueur si présent.
        if vals and vals[0] in (0, 1):
            vals = vals[1:]

        current["pos"] = vals

valid = [f for f in frames if f["pos"] and len(f["pos"]) >= 63]

def center_tori(pos):
    # 21 bodyparts * xyz = 63 valeurs
    pts = []
    for k in range(0, 63, 3):
        pts.append((pos[k], pos[k + 1], pos[k + 2]))

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

    ax, ay, az = center_tori(a["pos"])
    bx, by, bz = center_tori(b["pos"])

    dy = by - ay
    dx = bx - ax
    dz = bz - az

    windows.append({
        "start_idx": i,
        "end_idx": i + W,
        "start_frame": a["frame"],
        "end_frame": b["frame"],
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "speed_y": dy / max(1, b["frame"] - a["frame"]),
        "start_center": [ax, ay, az],
        "end_center": [bx, by, bz],
    })

windows.sort(key=lambda w: w["dy"], reverse=True)
OUT.write_text(json.dumps(windows[:120], indent=2), encoding="utf-8")

print(f"[OK] frames valid: {len(valid)}")
print(f"[OK] écrit: {OUT}")
print("Top 30 fenêtres par déplacement Y:")
for w in windows[:30]:
    print(
        f"{w['start_frame']}→{w['end_frame']} "
        f"dy={w['dy']:.3f} dx={w['dx']:.3f} "
        f"z={w['start_center'][2]:.2f}->{w['end_center'][2]:.2f}"
    )
