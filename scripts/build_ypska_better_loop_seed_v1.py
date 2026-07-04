#!/usr/bin/env python3
import json
import re
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"

OUT = ROOT / "evolution/trajectory_seed_v4_3_xioi_commands.json"
BACKUP = ROOT / "evolution/trajectory_seed_v4_3_xioi_commands.backup_before_looped_127_170.json"

LAUNCH_END = 126
LOOP_SRC_START = 128
LOOP_SRC_END = 170
LOOP_DST_START = 126
MAX_FRAME = 428

def find_ypska_rpl():
    for base in [ROOT / "replays_raw", ROOT / "datasets", ROOT]:
        if not base.exists():
            continue
        hits = [p for p in base.rglob("*.rpl") if "ypska" in p.name.lower()]
        if hits:
            hits.sort(key=lambda p: len(str(p)))
            return hits[0]
    raise FileNotFoundError("Aucun .rpl YpSkA trouvé.")

def parse_rpl_commands(path):
    pairs_by_frame = {}
    current_frame = None

    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()

        m = re.match(r"^FRAME\s+(\d+)", line, re.I)
        if m:
            current_frame = int(m.group(1))
            continue

        m = re.match(r"^JOINT\s+\d+\s*;\s*(.*)$", line, re.I)
        if m and current_frame is not None:
            nums = [int(x) for x in re.findall(r"-?\d+", m.group(1))]
            pairs = []
            for i in range(0, len(nums) - 1, 2):
                j, v = nums[i], nums[i + 1]
                if 0 <= j <= 19 and v in (1, 2, 3, 4):
                    pairs.append([j, v])
            if pairs:
                pairs_by_frame[current_frame] = pairs

    return [{"frame": f, "pairs": pairs_by_frame[f]} for f in sorted(pairs_by_frame)]

def main():
    rpl = find_ypska_rpl()
    commands = parse_rpl_commands(rpl)

    if OUT.exists() and not BACKUP.exists():
        BACKUP.write_text(OUT.read_text())

    out = []

    # Launch gelé 0 -> 126
    for cmd in commands:
        f = int(cmd["frame"])
        if 0 <= f < LAUNCH_END:
            out.append({"frame": f, "pairs": cmd["pairs"]})

    # Loop 127 -> 170 répétée de 126 à 428
    loop_cmds = [
        cmd for cmd in commands
        if LOOP_SRC_START <= int(cmd["frame"]) <= LOOP_SRC_END
    ]

    loop_len = LOOP_SRC_END - LOOP_SRC_START + 1
    dst = LOOP_DST_START

    while dst < MAX_FRAME:
        for cmd in loop_cmds:
            src_f = int(cmd["frame"])
            new_f = dst + (src_f - LOOP_SRC_START)
            if new_f >= MAX_FRAME:
                break
            out.append({"frame": new_f, "pairs": cmd["pairs"]})
        dst += loop_len

    out.sort(key=lambda c: int(c["frame"]))

    agent = {
        "name": "trajectory_seed_v4_3_ypska_launch_0_126_loop_128_170",
        "branch": "walk_trajectory_v4_3",
        "source_rpl": str(rpl),
        "launch": [0, LAUNCH_END],
        "loop_source": [LOOP_SRC_START, LOOP_SRC_END],
        "loop_remapped_to": [LOOP_DST_START, MAX_FRAME],
        "loop_repeated": True,
        "freeze_until": LAUNCH_END,
        "loop_length": MAX_FRAME,
        "commands": out,
    }

    OUT.write_text(json.dumps(agent, indent=2))
    print("✅ Seed loopé écrit:", OUT)
    print("RPL:", rpl)
    print("Loop:", LOOP_SRC_START, "->", LOOP_SRC_END)
    print("Commands:", len(out))

if __name__ == "__main__":
    main()
