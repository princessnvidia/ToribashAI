#!/usr/bin/env python3
"""
generate_xioi_loop_phase_calibration_v50_1.py
ToribashAI / walk_xioi

V50.1: micro calibration around the best V50 loop length.
Best reported by Vio: len260, still with a tiny delay.
This generates short RPL candidates with loop lengths around 260:
  256, 258, 260, 262, 264

It tries to preserve the trusted source replay context and only builds visual test
replays for selecting the best phase/loop length.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
STEAM_TORIBASH = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
REPLAY_ROOT = STEAM_TORIBASH / "replay"
REPLAY_PARKOUR = REPLAY_ROOT / "parkour"

# Preferred base from the phase work. Fall back to the known assassin/Xioi source.
SOURCE_CANDIDATES = [
    OUT_DIR / "xioi_same_foot_loop_walk_v35_1_base.rpl",
    OUT_DIR / "xioi_assassin_template_loop_v48.rpl",
    OUT_DIR / "xioi_427_assassincreedhunter_v37.rpl",
    OUT_DIR / "xioi_master_final_v5_champion.rpl",
]

VERSION = "50.1"
FIGHT_PREFIX = "xioi_loop_phase_v50_1"
MOD_NAME = "Urban_Structure/assassincreedhunter.tbm"

# Loop region chosen from V49/V50 work.
LOOP_START = 485
BASE_LOOP_LEN = 260
LOOP_LENGTHS = [256, 258, 260, 262, 264]

# Keep replay test short and readable. 1 source intro + 4 loop cycles.
INTRO_UNTIL = 315
CYCLES = 4
TURNFRAMES = 5
MATCHFRAMES_EXTRA = 120

FRAME_RE = re.compile(r"^FRAME\s+(-?\d+)\s*;")
NEWGAME_RE = re.compile(r"^NEWGAME\s+0;")
FIGHT_RE = re.compile(r"^FIGHTNAME\s+0;")


def find_source() -> Path:
    for p in SOURCE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("No source replay found. Tried:\n" + "\n".join(str(p) for p in SOURCE_CANDIDATES))


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def split_blocks(lines: list[str]) -> tuple[list[str], list[tuple[int, list[str]]]]:
    header: list[str] = []
    blocks: list[tuple[int, list[str]]] = []
    current_frame: int | None = None
    current: list[str] = []

    for line in lines:
        m = FRAME_RE.match(line)
        if m:
            if current_frame is None:
                header = current if not header else header
            else:
                blocks.append((current_frame, current))
            current_frame = int(m.group(1))
            current = [line]
        else:
            current.append(line)

    if current_frame is None:
        header = current
    else:
        blocks.append((current_frame, current))

    return header, blocks


def block_map(blocks: list[tuple[int, list[str]]]) -> dict[int, list[str]]:
    return {fr: block for fr, block in blocks}


def rewrite_block_frame(block: list[str], new_frame: int) -> list[str]:
    out = list(block)
    out[0] = re.sub(r"^FRAME\s+-?\d+\s*;", f"FRAME {new_frame};", out[0])
    return out


def rewrite_header(header: list[str], fightname: str, matchframes: int) -> list[str]:
    out: list[str] = []
    has_fight = False
    has_newgame = False

    for line in header:
        if FIGHT_RE.match(line):
            out.append(f"FIGHTNAME 0; {fightname}")
            has_fight = True
        elif NEWGAME_RE.match(line):
            # Preserve most rules but force matchframes + known mod.
            parts = line.split()
            # Robust simple NEWGAME rewrite matching our common format:
            # NEWGAME 0;matchframes turnframes ... mod
            rest = line.split(";", 1)[1].strip() if ";" in line else ""
            vals = rest.split()
            if len(vals) >= 2:
                vals[0] = str(matchframes)
                vals[1] = str(TURNFRAMES)
                if vals:
                    if vals[-1].endswith(".tbm") or vals[-1] == "classic" or "/" in vals[-1]:
                        vals[-1] = MOD_NAME
                    else:
                        vals.append(MOD_NAME)
                out.append("NEWGAME 0;" + " ".join(vals))
            else:
                out.append(f"NEWGAME 0;{matchframes} {TURNFRAMES} 30 0 0 2 100 0 0 0 0 0 0 0 {MOD_NAME}")
            has_newgame = True
        else:
            out.append(line)

    if not has_fight:
        insert_at = min(5, len(out))
        out.insert(insert_at, f"FIGHTNAME 0; {fightname}")
    if not has_newgame:
        out.append(f"NEWGAME 0;{matchframes} {TURNFRAMES} 30 0 0 2 100 0 0 0 0 0 0 0 {MOD_NAME}")

    return out


def nearest_frame(frames: list[int], target: int) -> int | None:
    if not frames:
        return None
    return min(frames, key=lambda f: abs(f - target))


def make_candidate(source: Path, loop_len: int) -> Path:
    lines = read_lines(source)
    header, blocks = split_blocks(lines)
    bmap = block_map(blocks)
    frames = sorted(bmap)

    # Preserve source intro up to INTRO_UNTIL, using original blocks but renumbering compactly.
    output_blocks: list[list[str]] = []
    out_frame = 0

    intro_frames = [f for f in frames if 0 <= f <= INTRO_UNTIL]
    if not intro_frames:
        raise RuntimeError("No intro frames found in source replay.")

    for fr in intro_frames:
        output_blocks.append(rewrite_block_frame(bmap[fr], out_frame))
        out_frame += TURNFRAMES

    # Add repeated loop cycles from LOOP_START -> LOOP_START + loop_len.
    loop_end = LOOP_START + loop_len
    src_loop_frames = [f for f in frames if LOOP_START <= f <= loop_end]
    if len(src_loop_frames) < 5:
        # fallback: sample nearest turnframes if source has sparse weird frames
        src_loop_frames = []
        for t in range(LOOP_START, loop_end + 1, TURNFRAMES):
            nf = nearest_frame(frames, t)
            if nf is not None and nf not in src_loop_frames:
                src_loop_frames.append(nf)

    for _ in range(CYCLES):
        for fr in src_loop_frames:
            output_blocks.append(rewrite_block_frame(bmap[fr], out_frame))
            out_frame += TURNFRAMES

    fightname = f"{FIGHT_PREFIX}_len{loop_len}"
    matchframes = out_frame + MATCHFRAMES_EXTRA
    new_header = rewrite_header(header, fightname, matchframes)

    out_path = OUT_DIR / f"{fightname}.rpl"
    out_text = "\n".join(new_header).rstrip() + "\n\n" + "\n\n".join("\n".join(b) for b in output_blocks) + "\n"
    out_path.write_text(out_text, encoding="utf-8")
    return out_path


def copy_to_steam(path: Path) -> None:
    REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
    REPLAY_PARKOUR.mkdir(parents=True, exist_ok=True)
    for dst_dir in [REPLAY_ROOT, REPLAY_PARKOUR]:
        dst = dst_dir / path.name
        shutil.copy2(path, dst)
        dst.touch()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = find_source()
    print("Source:", source)
    print("Loop start:", LOOP_START)
    print("Lengths:", LOOP_LENGTHS)

    made = []
    for length in LOOP_LENGTHS:
        p = make_candidate(source, length)
        copy_to_steam(p)
        made.append(p)
        print("Made:", p.name)

    print("\nCopied to Steam replay root + parkour.")
    print("Test in this order:")
    for length in LOOP_LENGTHS:
        print(f"  {FIGHT_PREFIX}_len{length}")


if __name__ == "__main__":
    main()
