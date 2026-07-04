#!/usr/bin/env python3
"""
evolution_xioi_master_final_v5_loop_extension.py

V5 = loop extension, not mutation.

Goal:
  - Keep the current walking champion intact.
  - Do NOT mutate existing JOINT commands.
  - Extend the walk by copying a late good step segment and appending it.

Default parent:
  generated_replays/xioi_v30_23_mut.rpl

Outputs:
  generated_replays/xioi_master_final_v5_loop_XX.rpl
  copied to Toribash replay folder.

Usage:
  python3 scripts/evolution_xioi_master_final_v5_loop_extension.py
"""

from __future__ import annotations

import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
STEAM_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)

PARENT_CANDIDATES = [
    OUT_DIR / "xioi_master_final_v4_champion.rpl",
    OUT_DIR / "xioi_master_final_v3_champion.rpl",
    OUT_DIR / "xioi_v30_23_mut.rpl",
    OUT_DIR / "xioi_v29_champion.rpl",
    OUT_DIR / "xioi_source_template_v28.rpl",
]

PREFIX = "xioi_master_final_v5_loop"
POPULATION = 16

# Nothing before this is touched.
PROTECT_UNTIL_FRAME = 140

# Candidate step-loop windows. These are copied exactly then shifted later.
# If one window is bad for your champion, the population tries several nearby variants.
BASE_WINDOWS = [
    (70, 140),
    (80, 155),
    (90, 170),
    (100, 180),
    (110, 195),
    (120, 210),
]

# Append 1-4 repetitions depending candidate.
REPEAT_OPTIONS = [1, 2, 2, 3, 3, 4]

# Small gap before appending copied segment.
APPEND_GAPS = [0, 5, 10, 15, 20]

FRAME_RE = re.compile(r"^FRAME\s+(-?\d+)\s*;")


@dataclass
class FrameBlock:
    frame: int
    lines: list[str]


def choose_parent() -> Path:
    for p in PARENT_CANDIDATES:
        if p.exists():
            return p
    matches = sorted(OUT_DIR.glob("*v30_23*.rpl")) + sorted(OUT_DIR.glob("*v31*exact*.rpl"))
    if matches:
        return matches[0]
    raise FileNotFoundError("No Xioi champion/template replay found in generated_replays")


def split_replay(text: str) -> tuple[list[str], list[FrameBlock]]:
    lines = text.splitlines()
    header: list[str] = []
    blocks: list[FrameBlock] = []

    current: list[str] | None = None
    current_frame: int | None = None

    for line in lines:
        m = FRAME_RE.match(line.strip())
        if m:
            if current is not None and current_frame is not None:
                blocks.append(FrameBlock(current_frame, current))
            current_frame = int(m.group(1))
            current = [line]
        else:
            if current is None:
                header.append(line)
            else:
                current.append(line)

    if current is not None and current_frame is not None:
        blocks.append(FrameBlock(current_frame, current))

    blocks.sort(key=lambda b: b.frame)
    return header, blocks


def max_frame(blocks: list[FrameBlock]) -> int:
    return max((b.frame for b in blocks), default=0)


def copy_block_with_shift(block: FrameBlock, new_frame: int, tag: str) -> FrameBlock:
    new_lines: list[str] = []
    first = True
    for line in block.lines:
        if first and FRAME_RE.match(line.strip()):
            new_lines.append(f"FRAME {new_frame};")
            new_lines.append(f"# {tag} copied_from={block.frame}")
            first = False
        else:
            new_lines.append(line)
    return FrameBlock(new_frame, new_lines)


def make_variant(header: list[str], blocks: list[FrameBlock], idx: int) -> str:
    rng = random.Random(31000 + idx)

    # Keep every original block exactly.
    out_blocks = [FrameBlock(b.frame, list(b.lines)) for b in blocks]

    start, end = rng.choice(BASE_WINDOWS)
    repeats = rng.choice(REPEAT_OPTIONS)
    gap = rng.choice(APPEND_GAPS)

    segment = [b for b in blocks if start <= b.frame <= end]
    if not segment:
        segment = [b for b in blocks if b.frame >= PROTECT_UNTIL_FRAME][:12]
    if not segment:
        segment = blocks[-12:]

    seg_start = segment[0].frame
    seg_end = segment[-1].frame
    seg_len = max(5, seg_end - seg_start + 5)

    append_at = max_frame(blocks) + gap + 5

    # Copy the same successful segment; no mutation.
    for r in range(repeats):
        for b in segment:
            nf = append_at + r * seg_len + (b.frame - seg_start)
            out_blocks.append(copy_block_with_shift(b, nf, f"v5_loop idx={idx} rep={r+1}/{repeats}"))

    out_blocks.sort(key=lambda b: b.frame)

    # Update fight name, keep everything else including source context/POS/QAT intact.
    new_header = []
    for line in header:
        if line.startswith("FIGHTNAME"):
            new_header.append(f"FIGHTNAME 0; {PREFIX}_{idx:02d}")
        else:
            new_header.append(line)

    body: list[str] = []
    body.extend(new_header)
    if body and body[-1].strip():
        body.append("")
    for b in out_blocks:
        body.extend(b.lines)
        body.append("")
    return "\n".join(body).rstrip() + "\n"


def cleanup_old_v5() -> None:
    for p in OUT_DIR.glob(f"{PREFIX}_*.rpl"):
        p.unlink(missing_ok=True)
    for p in STEAM_REPLAY_DIR.glob(f"{PREFIX}_*.rpl"):
        p.unlink(missing_ok=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STEAM_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    parent = choose_parent()
    text = parent.read_text(encoding="utf-8", errors="replace")
    header, blocks = split_replay(text)

    if not blocks:
        raise RuntimeError(f"No FRAME blocks found in {parent}")

    cleanup_old_v5()

    # Also copy parent for visual comparison.
    parent_out = OUT_DIR / f"{PREFIX}_00_parent.rpl"
    parent_out.write_text(text, encoding="utf-8")
    shutil.copy2(parent_out, STEAM_REPLAY_DIR / parent_out.name)

    print("Parent:", parent)
    print("Frames:", len(blocks), "max_frame:", max_frame(blocks))
    print("Strategy: no JOINT mutation; append copied late-step loops only")

    for i in range(1, POPULATION + 1):
        out_text = make_variant(header, blocks, i)
        out_path = OUT_DIR / f"{PREFIX}_{i:02d}.rpl"
        out_path.write_text(out_text, encoding="utf-8")
        shutil.copy2(out_path, STEAM_REPLAY_DIR / out_path.name)
        print("made:", out_path.name)

    print("\nCopied to Toribash replay dir:", STEAM_REPLAY_DIR)
    print("Test xioi_master_final_v5_loop_00_parent first, then 01-16.")


if __name__ == "__main__":
    main()
