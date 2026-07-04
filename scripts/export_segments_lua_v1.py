#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path.home() / "Documents" / "ToribashAI"
SRC = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay/archive_xioi_history/parkour/xioi_stable_loop_v49_base.rpl"
OUT = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script/toribashai_segments_v1.lua"

SEGMENTS = {
    "launch_transition": (0, 1680),
    "walk_loop_a": (1680, 2320),
    "walk_loop_b": (2320, 2960),
}

def nums(s):
    return [int(x) for x in re.findall(r"-?\d+", s)]

frames = []
current = None
last = {j: 3 for j in range(20)}

for line in SRC.read_text(encoding="utf-8", errors="ignore").splitlines():
    line = line.strip()

    if line.startswith("FRAME "):
        n = nums(line)
        current = {"frame": n[0], "joints": last.copy()}
        frames.append(current)

    elif current and line.startswith("JOINT "):
        parts = line.split(";", 1)[1].strip().split()
        for i in range(0, len(parts) - 1, 2):
            j = int(parts[i])
            v = int(parts[i + 1])
            if 0 <= j < 20:
                last[j] = v
                current["joints"][j] = v

def lua_actions(start, end):
    chunk = [f for f in frames if start <= f["frame"] <= end]
    lines = ["{"]
    for f in chunk:
        arr = [f["joints"][j] for j in range(20)]
        lines.append("  {" + ", ".join(map(str, arr)) + "},")
    lines.append("}")
    return "\n".join(lines)

out = ["-- ToribashAI segments V1", "segments = {}"]

for name, (start, end) in SEGMENTS.items():
    out.append(f"segments['{name}'] = {lua_actions(start, end)}")

OUT.write_text("\n\n".join(out), encoding="utf-8")
print(f"[OK] écrit: {OUT}")
for name, (s, e) in SEGMENTS.items():
    print(name, s, e)
