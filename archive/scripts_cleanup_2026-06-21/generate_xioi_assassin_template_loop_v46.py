#!/usr/bin/env python3
"""
generate_xioi_assassin_template_loop_v46.py

V46 = template-safe loop extension.

Goal:
  - Keep the source Xioi assassin RPL intact until frame 315.
  - After frame 315, append repeated copies of the real cycle 70 -> 295.
  - Do NOT use GRU free generation.
  - Preserve complete RPL blocks: FRAME / POS / QAT / LINVEL / ANGVEL / JOINT.
  - Rename FIGHTNAME so Toribash UI shows the correct replay name.

This is meant to test whether the physical loop itself can be extended by copying
full replay state blocks, before asking the GRU to correct anything.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"

SOURCE_RPL = OUT_DIR / "xioi_427_assassincreedhunter_v37.rpl"

OUT_RPL = OUT_DIR / "xioi_assassin_template_loop_v46.rpl"
OUT_REF = OUT_DIR / "xioi_assassin_template_loop_v46_reference.json"

TORIBASH_ROOT = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
REPLAY_ROOT = TORIBASH_ROOT / "replay"
REPLAY_PARKOUR = REPLAY_ROOT / "parkour"

FIGHTNAME = "xioi_assassin_template_loop_v46"

SOURCE_UNTIL = 315
CYCLE_START = 70
CYCLE_END = 295
TARGET_END = 1200

FRAME_RE = re.compile(r"^FRAME\s+(-?\d+)\s*;")
ENGAGE_RE = re.compile(r"^ENGAGE\s+(\d+)\s*;")
NEWGAME_RE = re.compile(r"^NEWGAME\s+0;")
FIGHTNAME_RE = re.compile(r"^FIGHTNAME\s+0;")


def split_blocks(lines: list[str]) -> tuple[list[str], list[dict]]:
    header: list[str] = []
    blocks: list[dict] = []
    current: dict | None = None

    for line in lines:
        m = FRAME_RE.match(line)
        if m:
            if current is not None:
                blocks.append(current)
            current = {"frame": int(m.group(1)), "lines": [line]}
        else:
            if current is None:
                header.append(line)
            else:
                current["lines"].append(line)

    if current is not None:
        blocks.append(current)

    return header, blocks


def rename_header(header: list[str]) -> list[str]:
    out: list[str] = []
    has_fightname = False
    for line in header:
        if FIGHTNAME_RE.match(line):
            out.append(f"FIGHTNAME 0; {FIGHTNAME}")
            has_fightname = True
        else:
            out.append(line)
    if not has_fightname:
        insert_at = 5 if len(out) > 5 else len(out)
        out.insert(insert_at, f"FIGHTNAME 0; {FIGHTNAME}")
    return out


def set_block_frame(block: dict, new_frame: int) -> dict:
    lines = list(block["lines"])
    old_first = lines[0]
    if FRAME_RE.match(old_first):
        # Keep suffix after ';' if present.
        suffix = old_first.split(";", 1)[1] if ";" in old_first else ""
        lines[0] = f"FRAME {new_frame};{suffix}"
    else:
        lines.insert(0, f"FRAME {new_frame}; 0 0 0 0")
    return {"frame": new_frame, "source_frame": block["frame"], "lines": lines}


def main() -> None:
    if not SOURCE_RPL.exists():
        raise FileNotFoundError(f"Missing source RPL: {SOURCE_RPL}")

    lines = SOURCE_RPL.read_text(encoding="utf-8", errors="ignore").splitlines()
    header, blocks = split_blocks(lines)
    header = rename_header(header)

    by_frame = {b["frame"]: b for b in blocks}
    frames_sorted = sorted(by_frame)

    source_blocks = [by_frame[f] for f in frames_sorted if f <= SOURCE_UNTIL]
    cycle_frames = [f for f in frames_sorted if CYCLE_START <= f <= CYCLE_END]
    cycle_blocks = [by_frame[f] for f in cycle_frames]

    if not source_blocks:
        raise RuntimeError("No source blocks selected")
    if len(cycle_blocks) < 4:
        raise RuntimeError(f"Cycle too small: {len(cycle_blocks)} blocks")

    # Estimate cycle length using the actual frame span.
    cycle_len = CYCLE_END - CYCLE_START
    output_blocks: list[dict] = []
    output_blocks.extend({"frame": b["frame"], "source_frame": b["frame"], "lines": b["lines"]} for b in source_blocks)

    # Append the full cycle repeatedly, preserving relative frame offsets.
    base = SOURCE_UNTIL + 5
    cycle_index = 0
    while base <= TARGET_END:
        for b in cycle_blocks:
            rel = b["frame"] - CYCLE_START
            new_frame = base + rel
            if new_frame > TARGET_END:
                break
            if new_frame <= SOURCE_UNTIL:
                continue
            output_blocks.append(set_block_frame(b, new_frame))
        cycle_index += 1
        base = SOURCE_UNTIL + 5 + cycle_index * cycle_len

    # Deduplicate by frame, last wins, and sort.
    dedup = {b["frame"]: b for b in output_blocks}
    output_blocks = [dedup[f] for f in sorted(dedup)]

    out_lines: list[str] = []
    out_lines.extend(header)
    if out_lines and out_lines[-1].strip() != "":
        out_lines.append("")

    for b in output_blocks:
        out_lines.extend(b["lines"])
        if out_lines and out_lines[-1].strip() != "":
            out_lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_RPL.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")

    summary = {
        "version": 46,
        "mode": "template_safe_cycle_copy",
        "source_rpl": str(SOURCE_RPL),
        "output_rpl": str(OUT_RPL),
        "fightname": FIGHTNAME,
        "source_until": SOURCE_UNTIL,
        "cycle_start": CYCLE_START,
        "cycle_end": CYCLE_END,
        "target_end": TARGET_END,
        "source_block_count": len(source_blocks),
        "cycle_block_count": len(cycle_blocks),
        "output_block_count": len(output_blocks),
        "output_frame_min": output_blocks[0]["frame"] if output_blocks else None,
        "output_frame_max": output_blocks[-1]["frame"] if output_blocks else None,
        "cycles_appended": cycle_index,
    }
    OUT_REF.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
    REPLAY_PARKOUR.mkdir(parents=True, exist_ok=True)
    for dst_dir in [REPLAY_ROOT, REPLAY_PARKOUR]:
        dst = dst_dir / OUT_RPL.name
        shutil.copy2(OUT_RPL, dst)
        dst.touch()
        print("Copied to:", dst)

    print("Wrote:", OUT_RPL)
    print("Reference:", OUT_REF)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
