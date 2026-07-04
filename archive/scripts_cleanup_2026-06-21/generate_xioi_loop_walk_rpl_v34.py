#!/usr/bin/env python3
"""
generate_xioi_loop_walk_rpl_v34.py

V34 = build a "learned walking loop" RPL template from the current best Xioi-style
walking champion.

Goal:
  - keep the faithful RPL physics context (POS/QAT/LINVEL/ANGVEL/etc.)
  - preserve the launch / first steps
  - extend the walking by copying a chosen step-loop segment several times
  - write clean FIGHTNAME names so Toribash UI can distinguish candidates
  - export a JSON reference map for future evolution / GRU training

This is intentionally RPL-first, not Lua-live. The RPL is the source of truth.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path.home() / "Documents" / "ToribashAI"
GEN_DIR = ROOT / "generated_replays"
OUT_DIR = GEN_DIR
STEAM_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)

# Best current known walking champion. The script will fall back if needed.
PARENT_CANDIDATES = [
    GEN_DIR / "xioi_master_final_v5_champion.rpl",
    GEN_DIR / "xioi_v33_champion.rpl",
    GEN_DIR / "xioi_master_final_v5_loop_10.rpl",
    GEN_DIR / "xioi_v30_23_mut.rpl",
]

OUT_RPL = OUT_DIR / "xioi_loop_walk_learned_v34.rpl"
OUT_JSON = OUT_DIR / "xioi_loop_walk_learned_v34_reference.json"

# These defaults are conservative. If the loop starts/ends at the wrong moment,
# edit LOOP_START/LOOP_END and rerun.
LAUNCH_LOCK_END = 150
LOOP_START = 120
LOOP_END = 260
LOOP_REPEATS = 4
FRAME_STEP = 5

# Candidate alternatives generated around the loop segment to let you visually pick.
MAKE_VARIANTS = True
VARIANT_SEGMENTS = [
    (105, 235),
    (115, 255),
    (120, 260),
    (130, 270),
    (140, 285),
]
VARIANT_REPEATS = [3, 4, 5]

FRAME_RE = re.compile(r"^FRAME\s+(-?\d+)\s*;")
FIGHT_RE = re.compile(r"^FIGHTNAME\s+0;")


@dataclass
class FrameBlock:
    frame: int
    lines: list[str]


def find_parent() -> Path:
    for p in PARENT_CANDIDATES:
        if p.exists():
            return p
    found = sorted(GEN_DIR.glob("*v33*g*_c*.rpl")) + sorted(GEN_DIR.glob("*v5*champion*.rpl"))
    if found:
        return found[-1]
    raise FileNotFoundError(
        "No parent RPL found. Expected one of:\n" + "\n".join(str(p) for p in PARENT_CANDIDATES)
    )


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def split_rpl(lines: list[str]) -> tuple[list[str], list[FrameBlock]]:
    header: list[str] = []
    frames: list[FrameBlock] = []
    current: list[str] | None = None
    current_frame: int | None = None

    for line in lines:
        m = FRAME_RE.match(line.strip())
        if m:
            if current is not None and current_frame is not None:
                frames.append(FrameBlock(current_frame, current))
            current_frame = int(m.group(1))
            current = [line]
        else:
            if current is None:
                header.append(line)
            else:
                current.append(line)

    if current is not None and current_frame is not None:
        frames.append(FrameBlock(current_frame, current))

    frames.sort(key=lambda b: b.frame)
    return header, frames


def set_fightname(header: list[str], name: str) -> list[str]:
    out: list[str] = []
    done = False
    for line in header:
        if FIGHT_RE.match(line):
            out.append(f"FIGHTNAME 0; {name}")
            done = True
        else:
            out.append(line)
    if not done:
        # Put it after VERSION if possible, otherwise near the top.
        inserted = False
        newer: list[str] = []
        for line in out:
            newer.append(line)
            if line.startswith("VERSION "):
                newer.append(f"FIGHTNAME 0; {name}")
                inserted = True
        out = newer
        if not inserted:
            out.insert(min(5, len(out)), f"FIGHTNAME 0; {name}")
    return out


def frame_block_with_new_frame(block: FrameBlock, new_frame: int, tag: str | None = None) -> list[str]:
    lines = list(block.lines)
    lines[0] = f"FRAME {new_frame};"
    if tag:
        # Keep comments useful but avoid exploding repeated comments.
        lines.insert(1, f"# {tag} source_frame={block.frame}")
    return lines


def nearest_blocks(frames: list[FrameBlock], start: int, end: int) -> list[FrameBlock]:
    return [b for b in frames if start <= b.frame <= end]


def make_looped_frames(
    frames: list[FrameBlock],
    loop_start: int,
    loop_end: int,
    repeats: int,
    launch_lock_end: int,
    frame_step: int,
) -> list[list[str]]:
    if loop_end <= loop_start:
        raise ValueError("loop_end must be greater than loop_start")

    prefix = [b for b in frames if b.frame < loop_start]
    loop = nearest_blocks(frames, loop_start, loop_end)
    if len(loop) < 4:
        raise ValueError(f"Loop segment too small: {len(loop)} frames between {loop_start}-{loop_end}")

    out: list[list[str]] = []

    # Copy the original launch / early walk exactly.
    for b in prefix:
        tag = "v34_launch_locked" if b.frame <= launch_lock_end else "v34_preloop"
        out.append(frame_block_with_new_frame(b, b.frame, tag))

    # Repeat selected loop. We preserve the relative frame spacing of the source.
    next_frame = max((b.frame for b in prefix), default=0) + frame_step
    loop_base = loop[0].frame
    last_written = next_frame - frame_step

    for r in range(repeats):
        for b in loop:
            rel = b.frame - loop_base
            nf = next_frame + rel
            if nf <= last_written:
                nf = last_written + frame_step
            out.append(frame_block_with_new_frame(b, nf, f"v34_loop_repeat={r+1}"))
            last_written = nf
        next_frame = last_written + frame_step

    return out


def write_rpl(path: Path, header: list[str], frame_lines: Iterable[list[str]], fightname: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header2 = set_fightname(header, fightname)
    lines: list[str] = []
    lines.extend(header2)
    if lines and lines[-1] != "":
        lines.append("")
    for block in frame_lines:
        lines.extend(block)
        if lines and lines[-1] != "":
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def extract_reference(path: Path, source: Path, loop_start: int, loop_end: int, repeats: int) -> None:
    lines = read_lines(path)
    _, frames = split_rpl(lines)
    ref = {
        "name": path.stem,
        "version": 34,
        "description": "Loop-walking RPL reference built from the best Xioi-style champion. Use this as the walking-map JSON for future proximity evolution and GRU dataset building.",
        "source_parent": str(source),
        "rpl": str(path),
        "loop_start": loop_start,
        "loop_end": loop_end,
        "loop_repeats": repeats,
        "frames": [],
    }
    for b in frames:
        joints: list[list[int]] = []
        for line in b.lines:
            if line.startswith("JOINT 0;"):
                # Accept multi-pair JOINT lines too.
                parts = line.split(";", 1)[1].strip().split()
                for i in range(0, len(parts) - 1, 2):
                    try:
                        joints.append([int(parts[i]), int(parts[i + 1])])
                    except ValueError:
                        pass
        ref["frames"].append({"frame": b.frame, "joint_pairs": joints})
    path_json = OUT_JSON if path == OUT_RPL else OUT_DIR / f"{path.stem}_reference.json"
    path_json.write_text(json.dumps(ref, indent=2), encoding="utf-8")
    print("reference:", path_json)


def copy_to_steam(paths: Iterable[Path]) -> None:
    STEAM_REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    for p in paths:
        shutil.copy2(p, STEAM_REPLAY_DIR / p.name)
        print("copied:", STEAM_REPLAY_DIR / p.name)


def main() -> None:
    parent = find_parent()
    print("parent:", parent)
    header, frames = split_rpl(read_lines(parent))
    print("frames:", len(frames), "min", min(b.frame for b in frames), "max", max(b.frame for b in frames))

    made: list[Path] = []

    main_frames = make_looped_frames(frames, LOOP_START, LOOP_END, LOOP_REPEATS, LAUNCH_LOCK_END, FRAME_STEP)
    write_rpl(OUT_RPL, header, main_frames, OUT_RPL.stem)
    made.append(OUT_RPL)
    extract_reference(OUT_RPL, parent, LOOP_START, LOOP_END, LOOP_REPEATS)
    print("made:", OUT_RPL)

    if MAKE_VARIANTS:
        idx = 1
        for start, end in VARIANT_SEGMENTS:
            for repeats in VARIANT_REPEATS:
                try:
                    variant_frames = make_looped_frames(frames, start, end, repeats, LAUNCH_LOCK_END, FRAME_STEP)
                except Exception as e:
                    print("skip variant", start, end, repeats, e)
                    continue
                out = OUT_DIR / f"xioi_loop_walk_v34_s{start}_e{end}_r{repeats}_c{idx:02d}.rpl"
                write_rpl(out, header, variant_frames, out.stem)
                extract_reference(out, parent, start, end, repeats)
                made.append(out)
                idx += 1
                print("made:", out.name)

    copy_to_steam(made)
    print("\nDone. Test in Toribash replay UI. The main candidate is:")
    print(" ", OUT_RPL.name)
    print("If one variant loops better, use its *_reference.json as the next walking map.")


if __name__ == "__main__":
    main()
