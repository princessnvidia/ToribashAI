#!/usr/bin/env python3
from pathlib import Path
import re

ROOT = Path.home() / "Documents" / "ToribashAI"
SRC = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay/archive_xioi_history/parkour/xioi_stable_loop_v49_base.rpl"
OUT_DIR = ROOT / "generated_replays" / "skills"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLIPS = {
    "launch_v2_fullphysics": (0, 320),

    "macro_launch_walk_0_840": (0, 840),
    "walk_loop_stable_1680_2320": (1680, 2320),
    "walk_loop_stable_2320_2960": (2320, 2960),
    "transition_320_1680": (320, 1680),

    "walk_v2_fullphysics_520_840": (520, 840),
    "walk_v2_fullphysics_1560_1880": (1560, 1880),
    "walk_v2_fullphysics_3920_4240": (3920, 4240),
    "walk_v2_fullphysics_6800_7120": (6800, 7120),
}

lines = SRC.read_text(encoding="utf-8", errors="ignore").splitlines()

header = []
blocks = []
current = []

for line in lines:
    if line.startswith("FRAME "):
        if current:
            blocks.append(current)
        current = [line]
    else:
        if current:
            current.append(line)
        else:
            header.append(line)

if current:
    blocks.append(current)


def frame_num(block):
    m = re.search(r"FRAME\s+(-?\d+)", block[0])
    return int(m.group(1))


for name, (start, end) in CLIPS.items():
    selected = [b for b in blocks if start <= frame_num(b) <= end]

    out = OUT_DIR / f"{name}.rpl"

    new_header = []
    for h in header:
        if h.startswith("FIGHTNAME 0;"):
            new_header.append(f"FIGHTNAME 0; ToribashAI skill {name}")
        elif h.startswith("AUTHOR 0;"):
            new_header.append("AUTHOR 0; ToribashAI")
        else:
            new_header.append(h)

    out_lines = list(new_header)

    for b in selected:
        old_frame = frame_num(b)
        new_frame = old_frame - start

        for line in b:
            if line.startswith("FRAME "):
                line = re.sub(r"FRAME\s+-?\d+", f"FRAME {new_frame}", line)
            out_lines.append(line)

    out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[OK] {name}: frames={len(selected)} start={start} end={end} -> {out}")
