#!/usr/bin/env python3
import json
import re
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"

RPL = ROOT / "replays_raw/parkour_candidate/#Xioi#Pk - YpSkA.rpl"
OUT = ROOT / "evolution/walk_mechanic_xioi_v1.json"

MAX_FRAME = 427

FRAME_RE = re.compile(r"^FRAME\s+(\d+);")
JOINT_RE = re.compile(r"^JOINT\s+0;\s*(.*)$")


def fix(v):
    v = int(v)
    if v < 1:
        return 3
    if v > 4:
        return 4
    return v


def main():
    current_frame = None
    commands = []

    for line in RPL.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()

        m = FRAME_RE.match(line)
        if m:
            current_frame = int(m.group(1))
            continue

        m = JOINT_RE.match(line)
        if m and current_frame is not None and current_frame <= MAX_FRAME:
            nums = [int(x) for x in m.group(1).split()]
            pairs = []

            for i in range(0, len(nums) - 1, 2):
                pairs.append([nums[i], fix(nums[i + 1])])

            commands.append({
                "frame": current_frame,
                "pairs": pairs,
            })

    mechanic = {
        "name": "xioi_walk_mechanic_v1",
        "source_rpl": str(RPL),
        "max_frame": MAX_FRAME,
        "loop_length": MAX_FRAME + 1,
        "command_count": len(commands),
        "description": "Mémoire de marche Xioi: commandes JOINT exactes jusqu'à frame 427, destinées à être bouclées et mutées légèrement.",
        "allowed_mutation": {
            "type": "small_command_mutation",
            "max_mutation_rate": 0.015,
            "allowed_values": [1, 2, 3, 4],
            "protect_arms": True,
            "protect_torso": True
        },
        "commands": commands
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(mechanic, indent=2), encoding="utf-8")

    print("Mémoire créée:", OUT)
    print("Commands:", len(commands))
    print("Loop length:", MAX_FRAME + 1)


if __name__ == "__main__":
    main()
