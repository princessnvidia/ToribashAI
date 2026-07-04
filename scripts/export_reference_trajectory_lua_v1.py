#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
SRC = ROOT / "evolution" / "reference_trajectory_v1.json"
OUT = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script/toribashai_reference_trajectory_v1.lua"

data = json.loads(SRC.read_text())

lines = ["-- ToribashAI reference trajectory V1", "reference_trajectory = {}"]

for name, seg in data["segments"].items():
    lines.append(f'reference_trajectory["{name}"] = {{')
    for p in seg["points"]:
        lines.append(
            f'  {{frame={p["frame"]}, x={p["x"]:.6f}, y={p["y"]:.6f}, z={p["z"]:.6f}}},'
        )
    lines.append("}")

OUT.write_text("\n".join(lines), encoding="utf-8")

print(f"[OK] écrit: {OUT}")
for name, seg in data["segments"].items():
    print(name, len(seg["points"]))
