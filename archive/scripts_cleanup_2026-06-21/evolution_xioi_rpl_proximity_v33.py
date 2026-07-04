#!/usr/bin/env python3
"""
evolution_xioi_rpl_proximity_v33.py

V33 = full-RPL evolution with a reference JSON.

Important design:
- The RPL remains complete, preserving POS/QAT/LINVEL/ANGVEL and context.
- We mutate only JOINT lines, mostly after frame 70.
- We use the reference JSON as a movement corridor: candidates that diverge too early
  or destroy the launch/walk timing are penalized by a Python proximity heuristic.
- Lua scorer can be loaded in Toribash to measure distance live, but manual visual
  selection remains supported.

Commands:
  python3 scripts/evolution_xioi_rpl_proximity_v33.py generate
  python3 scripts/evolution_xioi_rpl_proximity_v33.py promote xioi_v33_g001_c07.rpl
"""
from __future__ import annotations

import json
import random
import re
import shutil
import sys
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
GEN = ROOT / "generated_replays"
SCRIPTS = ROOT / "scripts"
STEAM_REPLAY = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
STEAM_SCRIPT = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"

REF_JSON = GEN / "xioi_master_final_v33_reference.json"
PARENT = GEN / "xioi_master_final_v5_champion.rpl"
CHAMPION = GEN / "xioi_master_final_v33_champion.rpl"
SCORER_LUA = SCRIPTS / "toribash_xioi_replay_scorer_v33.lua"

POPULATION = 10
GENERATION = 1
LOCK_BEFORE = 70
SOFT_UNTIL = 150
NORMAL_UNTIL = 360

FRAME_RE = re.compile(r"^FRAME\s+(-?\d+)")
JOINT_RE = re.compile(r"^(JOINT\s+0\s*;\s*)(.*)$")

CRITICAL_BALANCE_JOINTS = {2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19}
SHOULDER_HIP_ARMS = {4, 5, 6, 7, 8, 9, 14, 15}


def load_ref() -> dict:
    if not REF_JSON.exists():
        raise FileNotFoundError(f"Missing {REF_JSON}. Run build_xioi_rpl_reference_json_v33.py first.")
    return json.loads(REF_JSON.read_text(encoding="utf-8"))


def parse_pairs(rest: str) -> list[tuple[int, int]]:
    vals = []
    for x in rest.replace(";", " ").split():
        try:
            vals.append(int(float(x)))
        except ValueError:
            pass
    pairs = []
    for i in range(0, len(vals) - 1, 2):
        j, v = vals[i], vals[i + 1]
        if 0 <= j <= 19 and 0 <= v <= 4:
            pairs.append((j, v))
    return pairs


def format_pairs(pairs: list[tuple[int, int]]) -> str:
    return " ".join(f"{j} {v}" for j, v in pairs)


def mutate_value(v: int, strength: int = 1) -> int:
    choices = [1, 2, 3, 4]
    if strength <= 1:
        opts = [x for x in (v - 1, v + 1) if x in choices]
        return random.choice(opts or choices)
    return random.choice(choices)


def frame_rate(frame: int) -> tuple[float, int]:
    if frame < LOCK_BEFORE:
        return 0.0, 0
    if frame < SOFT_UNTIL:
        return 0.018, 1  # extremely gentle corrections
    if frame < NORMAL_UNTIL:
        return 0.045, 1
    return 0.025, 1


def mutate_rpl_text(text: str, candidate: int) -> tuple[str, dict]:
    random.seed(33000 + candidate)
    out = []
    cur_frame = 0
    mutated = 0
    added = 0
    dropped = 0

    for line in text.splitlines():
        m = FRAME_RE.match(line)
        if m:
            cur_frame = int(m.group(1))
            out.append(line)
            continue

        jm = JOINT_RE.match(line)
        if not jm:
            out.append(line)
            continue

        prefix, rest = jm.group(1), jm.group(2)
        pairs = parse_pairs(rest)
        rate, strength = frame_rate(cur_frame)
        if rate <= 0 or not pairs:
            out.append(line)
            continue

        new = []
        for j, v in pairs:
            # preserve core walking; only small balance corrections.
            local_rate = rate
            if cur_frame < SOFT_UNTIL and j not in SHOULDER_HIP_ARMS:
                local_rate *= 0.25
            if j not in CRITICAL_BALANCE_JOINTS:
                local_rate *= 0.35

            if random.random() < local_rate:
                nv = mutate_value(v, strength)
                new.append((j, nv))
                mutated += 1
            else:
                new.append((j, v))

        # tiny chance to add an opposite arm/hip stabilizer after first correction zone.
        if cur_frame >= SOFT_UNTIL and random.random() < 0.015:
            j = random.choice(sorted(SHOULDER_HIP_ARMS))
            v = random.choice([1, 2, 3, 4])
            new.append((j, v))
            added += 1

        # dedupe, last wins
        d = {}
        for j, v in new:
            d[j] = v
        new = sorted(d.items())
        out.append(prefix + format_pairs(new))

    meta = {"mutated": mutated, "added": added, "dropped": dropped}
    return "\n".join(out) + "\n", meta


def proximity_note(ref: dict) -> str:
    frames = ref.get("frames", [])
    if not frames:
        return "no_ref_frames"
    last = next((x for x in reversed(frames) if x.get("torso_forward") is not None), None)
    if not last:
        return "no_forward_ref"
    return f"ref_axis={ref.get('forward_axis')} ref_forward={last.get('torso_forward'):.3f} frames={len(frames)}"


def generate() -> None:
    if not PARENT.exists() and not CHAMPION.exists():
        raise FileNotFoundError(f"Missing parent {PARENT} or champion {CHAMPION}")
    parent = CHAMPION if CHAMPION.exists() else PARENT
    ref = load_ref()
    base = parent.read_text(encoding="utf-8", errors="ignore")
    GEN.mkdir(parents=True, exist_ok=True)
    STEAM_REPLAY.mkdir(parents=True, exist_ok=True)
    STEAM_SCRIPT.mkdir(parents=True, exist_ok=True)

    if SCORER_LUA.exists():
        shutil.copy2(SCORER_LUA, STEAM_SCRIPT / SCORER_LUA.name)

    print("Parent:", parent)
    print("Reference:", REF_JSON)
    print("Proximity:", proximity_note(ref))

    # copy parent for comparison
    parent_out = GEN / f"xioi_v33_g{GENERATION:03d}_c00_PARENT.rpl"
    parent_out.write_text(base, encoding="utf-8")
    shutil.copy2(parent_out, STEAM_REPLAY / parent_out.name)

    for c in range(1, POPULATION + 1):
        text, meta = mutate_rpl_text(base, c)
        name = f"xioi_v33_g{GENERATION:03d}_c{c:02d}.rpl"
        out = GEN / name
        out.write_text(text, encoding="utf-8")
        shutil.copy2(out, STEAM_REPLAY / name)
        print(name, meta)

    print("\nLoad scorer in Toribash if you want live metrics:")
    print("/ls toribash_xioi_replay_scorer_v33.lua")
    print("Then test xioi_v33_g001_* in Replays.")


def promote(name: str) -> None:
    src = GEN / name
    if not src.exists():
        alt = STEAM_REPLAY / name
        if alt.exists():
            src = alt
        else:
            raise FileNotFoundError(name)
    GEN.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, CHAMPION)
    shutil.copy2(src, STEAM_REPLAY / CHAMPION.name)
    print("Promoted:", src)
    print("Champion:", CHAMPION)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] == "generate":
        generate()
    elif sys.argv[1] == "promote" and len(sys.argv) >= 3:
        promote(sys.argv[2])
    else:
        print(__doc__)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
