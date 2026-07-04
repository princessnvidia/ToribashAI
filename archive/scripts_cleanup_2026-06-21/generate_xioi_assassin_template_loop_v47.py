#!/usr/bin/env python3
"""
generate_xioi_assassin_template_loop_v47.py

V47 = template-safe translated loop extension.

Why V47 exists:
  V46 proved the cycle joints are good, but the copied cycle used old absolute POS
  coordinates. At the loop junction, Tori could snap / drift back spatially.

V47 keeps the source replay intact until frame 315, then copies the real walking
cycle with a Y translation so each new cycle starts where the previous one ended.
It also generates several phase variants so we can skip the first bad raccord and
start directly on the most stable foot/pose.

Outputs copied to:
  Toribash/replay/
  Toribash/replay/parkour/
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
SOURCE_RPL = OUT_DIR / "xioi_427_assassincreedhunter_v37.rpl"

TORIBASH_ROOT = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
REPLAY_ROOT = TORIBASH_ROOT / "replay"
REPLAY_PARKOUR = REPLAY_ROOT / "replay/parkour" if False else REPLAY_ROOT / "parkour"

SOURCE_UNTIL = 315
CYCLE_START = 70
CYCLE_END = 295
TARGET_END = 1400

# Try several cycle entry points. If one phase starts on the correct planted foot,
# it should remove the visible first bad raccord.
PHASES = [70, 90, 110, 130, 150, 170]

FRAME_RE = re.compile(r"^FRAME\s+(-?\d+)\s*;")
FIGHTNAME_RE = re.compile(r"^FIGHTNAME\s+0;")
POS_RE = re.compile(r"^(POS\s+(\d+)\s*;)(.*)$")


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


def rename_header(header: list[str], fightname: str) -> list[str]:
    out: list[str] = []
    has = False
    for line in header:
        if FIGHTNAME_RE.match(line):
            out.append(f"FIGHTNAME 0; {fightname}")
            has = True
        else:
            out.append(line)
    if not has:
        insert_at = 5 if len(out) > 5 else len(out)
        out.insert(insert_at, f"FIGHTNAME 0; {fightname}")
    return out


def parse_floats(s: str) -> list[float] | None:
    try:
        return [float(x) for x in s.strip().split()]
    except ValueError:
        return None


def pos_values_for_player(block: dict, player: str = "0") -> list[float] | None:
    for line in block["lines"]:
        m = POS_RE.match(line)
        if not m:
            continue
        if m.group(2) != player:
            continue
        vals = parse_floats(m.group(3))
        if vals and len(vals) >= 3:
            return vals
    return None


def mean_y(block: dict, player: str = "0") -> float | None:
    vals = pos_values_for_player(block, player)
    if not vals:
        return None
    ys = []
    for i in range(1, len(vals), 3):
        ys.append(vals[i])
    return sum(ys) / len(ys) if ys else None


def nearest_block_with_pos(frames: list[int], by_frame: dict[int, dict], target: int, direction: int = -1) -> dict:
    candidates = [f for f in frames if (f <= target if direction < 0 else f >= target)]
    candidates = sorted(candidates, reverse=(direction < 0))
    for f in candidates:
        b = by_frame[f]
        if mean_y(b) is not None:
            return b
    raise RuntimeError(f"No POS block found near frame {target}")


def set_block_frame_and_translate_y(block: dict, new_frame: int, y_offset: float, only_player: str = "0") -> dict:
    out_lines: list[str] = []
    for idx, line in enumerate(block["lines"]):
        if idx == 0 and FRAME_RE.match(line):
            suffix = line.split(";", 1)[1] if ";" in line else " 0 0 0 0"
            out_lines.append(f"FRAME {new_frame};{suffix}")
            continue

        m = POS_RE.match(line)
        if m and m.group(2) == only_player:
            vals = parse_floats(m.group(3))
            if vals and len(vals) >= 3:
                vals2 = list(vals)
                # POS is triplets x y z. Shift only Y, preserving the walk shape.
                for i in range(1, len(vals2), 3):
                    vals2[i] += y_offset
                packed = " ".join(f"{v:.8f}" for v in vals2)
                out_lines.append(f"{m.group(1)} {packed}")
                continue

        out_lines.append(line)

    return {
        "frame": new_frame,
        "source_frame": block["frame"],
        "y_offset": y_offset,
        "lines": out_lines,
    }


def ordered_cycle_frames(cycle_frames: list[int], phase: int) -> list[int]:
    # Start from the first real source frame >= phase, then wrap to the start.
    after = [f for f in cycle_frames if f >= phase]
    before = [f for f in cycle_frames if f < phase]
    if not after:
        return cycle_frames
    return after + before


def write_replay(fightname: str, header: list[str], output_blocks: list[dict], out_rpl: Path) -> None:
    out_lines: list[str] = []
    out_lines.extend(rename_header(header, fightname))
    if out_lines and out_lines[-1].strip():
        out_lines.append("")
    for b in output_blocks:
        out_lines.extend(b["lines"])
        if out_lines and out_lines[-1].strip():
            out_lines.append("")
    out_rpl.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    if not SOURCE_RPL.exists():
        raise FileNotFoundError(f"Missing source RPL: {SOURCE_RPL}")

    lines = SOURCE_RPL.read_text(encoding="utf-8", errors="ignore").splitlines()
    header, blocks = split_blocks(lines)
    by_frame = {b["frame"]: b for b in blocks}
    frames_sorted = sorted(by_frame)

    source_frames = [f for f in frames_sorted if f <= SOURCE_UNTIL]
    source_blocks = [{"frame": by_frame[f]["frame"], "source_frame": by_frame[f]["frame"], "y_offset": 0.0, "lines": by_frame[f]["lines"]} for f in source_frames]
    cycle_frames = [f for f in frames_sorted if CYCLE_START <= f <= CYCLE_END]
    if len(source_blocks) < 4:
        raise RuntimeError("Not enough source blocks")
    if len(cycle_frames) < 4:
        raise RuntimeError("Not enough cycle blocks")

    until_block = nearest_block_with_pos(frames_sorted, by_frame, SOURCE_UNTIL, direction=-1)
    start_block = nearest_block_with_pos(frames_sorted, by_frame, CYCLE_START, direction=1)
    end_block = nearest_block_with_pos(frames_sorted, by_frame, CYCLE_END, direction=-1)

    y_until = mean_y(until_block)
    y_start = mean_y(start_block)
    y_end = mean_y(end_block)
    assert y_until is not None and y_start is not None and y_end is not None

    cycle_delta_y = y_end - y_start
    base_align_y = y_until - y_start

    # Preserve the real frame gaps inside the chosen phase order by emitting blocks
    # every 5 frames after SOURCE_UNTIL. This avoids keeping old source frame gaps
    # after wrap-around.
    output_infos = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
    REPLAY_PARKOUR.mkdir(parents=True, exist_ok=True)

    for phase in PHASES:
        phase_frames = ordered_cycle_frames(cycle_frames, phase)
        fightname = f"xioi_assassin_template_loop_v47_phase{phase:03d}"
        out_rpl = OUT_DIR / f"{fightname}.rpl"

        output_blocks: list[dict] = list(source_blocks)
        new_frame = SOURCE_UNTIL + 5
        cycle_index = 0

        while new_frame <= TARGET_END:
            for src_frame in phase_frames:
                if new_frame > TARGET_END:
                    break
                # Continuous spatial translation: first cycle aligns to frame 315,
                # following cycles move by the original cycle's Y progress.
                y_offset = base_align_y + (cycle_index * cycle_delta_y)
                output_blocks.append(set_block_frame_and_translate_y(by_frame[src_frame], new_frame, y_offset))
                new_frame += 5
            cycle_index += 1

        dedup = {b["frame"]: b for b in output_blocks}
        output_blocks = [dedup[f] for f in sorted(dedup)]
        write_replay(fightname, header, output_blocks, out_rpl)

        for dst_dir in [REPLAY_ROOT, REPLAY_PARKOUR]:
            dst = dst_dir / out_rpl.name
            shutil.copy2(out_rpl, dst)
            dst.touch()

        output_infos.append({
            "fightname": fightname,
            "rpl": str(out_rpl),
            "phase": phase,
            "output_frames": len(output_blocks),
            "frame_max": output_blocks[-1]["frame"],
            "base_align_y": base_align_y,
            "cycle_delta_y": cycle_delta_y,
            "cycles_written": cycle_index,
        })
        print("Wrote:", out_rpl)

    summary = {
        "version": 47,
        "mode": "template_safe_cycle_copy_with_y_translation_and_phase_variants",
        "source_rpl": str(SOURCE_RPL),
        "source_until": SOURCE_UNTIL,
        "cycle_start": CYCLE_START,
        "cycle_end": CYCLE_END,
        "target_end": TARGET_END,
        "y_until": y_until,
        "y_start": y_start,
        "y_end": y_end,
        "base_align_y": base_align_y,
        "cycle_delta_y": cycle_delta_y,
        "outputs": output_infos,
        "hint": "Test phase150/170 first if the first raccord is still unstable; test phase070 if later loops lose the same-foot timing.",
    }
    summary_path = OUT_DIR / "xioi_assassin_template_loop_v47_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Summary:", summary_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
