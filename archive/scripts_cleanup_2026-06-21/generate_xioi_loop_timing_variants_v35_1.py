#!/usr/bin/env python3
"""
generate_xioi_loop_timing_variants_v35_1.py

V35.1: generate timing variants for the Xioi same-foot loop.
Goal: fix the second loop being slightly late for the right foot ground contact.

It reads the current champion RPL, keeps the launch / first successful walk section,
then appends repeated loop segments with different timing windows:
  - loop start a bit earlier/later
  - loop end shifted with same length
  - optional small Y-distance scale

The script keeps the full RPL physics lines (POS/QAT/LINVEL/ANGVEL/JOINT/etc.)
and only shifts POS Y coordinates for appended loops, so Toribash still sees a
complete replay, not actions-only.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
STEAM_REPLAY_ROOT = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)
STEAM_PARKOUR = STEAM_REPLAY_ROOT / "parkour"
STEAM_MY_REPLAYS = STEAM_REPLAY_ROOT / "my replays"

# Prefer the current best loop champion. Fall back to older champion if needed.
SOURCE_CANDIDATES = [
    OUT_DIR / "xioi_same_foot_loop_walk_v35.rpl",
    OUT_DIR / "xioi_master_final_v5_champion.rpl",
    OUT_DIR / "xioi_v30_23_mut.rpl",
]

# Current good-ish loop was 150 -> 230. We test phase shifts around it.
BASE_START = 150
BASE_END = 230
LOOP_REPEATS = 5

# Mostly earlier starts because the second loop was late on right-foot contact.
VARIANTS = [
    ("early15", 135, 215, 1.00),
    ("early10", 140, 220, 1.00),
    ("early05", 145, 225, 1.00),
    ("base",    150, 230, 1.00),
    ("late05",  155, 235, 1.00),
    ("early10_shortY", 140, 220, 0.92),
    ("early10_longY",  140, 220, 1.08),
    ("early05_shortY", 145, 225, 0.92),
    ("early05_longY",  145, 225, 1.08),
]

FRAME_RE = re.compile(r"^FRAME\s+(-?\d+)\s*;")
FIGHT_RE = re.compile(r"^FIGHTNAME\s+0\s*;")
POS_RE = re.compile(r"^(POS\s+0\s*;)(.*)$")


@dataclass
class FrameBlock:
    frame: int
    lines: list[str]


def find_source() -> Path:
    for p in SOURCE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("No source champion found. Looked for:\n" + "\n".join(str(p) for p in SOURCE_CANDIDATES))


def split_rpl(path: Path) -> tuple[list[str], list[FrameBlock]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header: list[str] = []
    frames: list[FrameBlock] = []
    current: list[str] | None = None
    current_frame: int | None = None

    for line in lines:
        m = FRAME_RE.match(line)
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
        insert_at = min(5, len(out))
        out.insert(insert_at, f"FIGHTNAME 0; {name}")
    return out


def shift_pos_y(line: str, y_shift: float) -> str:
    """Shift every Y coordinate in a POS 0; line by y_shift.

    Toribash POS lines are flat xyz triples. If the line is not parseable, keep it.
    """
    m = POS_RE.match(line)
    if not m:
        return line
    prefix, rest = m.group(1), m.group(2)
    parts = rest.strip().split()
    try:
        vals = [float(x) for x in parts]
    except ValueError:
        return line
    if len(vals) < 3:
        return line
    for i in range(1, len(vals), 3):
        vals[i] += y_shift
    return prefix + " " + " ".join(f"{v:.8f}" for v in vals)


def clone_block(block: FrameBlock, new_frame: int, y_shift: float, tag: str) -> FrameBlock:
    new_lines: list[str] = []
    for idx, line in enumerate(block.lines):
        if idx == 0 and FRAME_RE.match(line):
            new_lines.append(f"FRAME {new_frame};")
            new_lines.append(f"# loopcopy {tag} src_frame={block.frame} y_shift={y_shift:.4f}")
        else:
            new_lines.append(shift_pos_y(line, y_shift))
    return FrameBlock(new_frame, new_lines)


def get_root_y(block: FrameBlock) -> float | None:
    for line in block.lines:
        m = POS_RE.match(line)
        if not m:
            continue
        parts = m.group(2).strip().split()
        if len(parts) >= 2:
            try:
                return float(parts[1])
            except ValueError:
                return None
    return None


def nearest_frame(frames: list[FrameBlock], target: int) -> FrameBlock:
    return min(frames, key=lambda b: abs(b.frame - target))


def frame_range(frames: list[FrameBlock], start: int, end: int) -> list[FrameBlock]:
    return [b for b in frames if start <= b.frame <= end]


def write_rpl(path: Path, header: list[str], frames: Iterable[FrameBlock]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.extend(header)
    if lines and lines[-1].strip():
        lines.append("")
    for block in sorted(frames, key=lambda b: b.frame):
        lines.extend(block.lines)
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def copy_to_steam(path: Path) -> None:
    for dest_dir in [STEAM_REPLAY_ROOT, STEAM_PARKOUR, STEAM_MY_REPLAYS]:
        try:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest_dir / path.name)
        except Exception as exc:
            print(f"warning: could not copy to {dest_dir}: {exc}")


def build_variant(source: Path, header: list[str], frames: list[FrameBlock], label: str, start: int, end: int, y_scale: float) -> dict:
    loop_len = end - start
    if loop_len <= 0:
        raise ValueError(f"bad loop {start}->{end}")

    start_block = nearest_frame(frames, start)
    end_block = nearest_frame(frames, end)
    y0 = get_root_y(start_block)
    y1 = get_root_y(end_block)
    if y0 is None or y1 is None:
        # Fall back to V35 observed delta.
        base_delta_y = -5.1595
    else:
        base_delta_y = (y1 - y0) * y_scale

    # Keep launch + the first existing loop fully intact.
    output_frames: list[FrameBlock] = [b for b in frames if b.frame <= end]

    segment = [b for b in frames if start <= b.frame <= end]
    if not segment:
        raise ValueError(f"no segment for {start}->{end}")

    # Append loop copies. Skip source frame == start to avoid duplicate contact frame.
    for rep in range(1, LOOP_REPEATS + 1):
        y_shift = base_delta_y * rep
        frame_offset = end + (rep - 1) * loop_len - start
        for b in segment:
            if b.frame == start:
                continue
            new_frame = b.frame + frame_offset
            output_frames.append(clone_block(b, new_frame, y_shift, f"{label}_rep{rep}"))

    name = f"xioi_same_foot_loop_walk_v35_1_{label}"
    out_path = OUT_DIR / f"{name}.rpl"
    out_header = set_fightname(header, name)
    write_rpl(out_path, out_header, output_frames)
    copy_to_steam(out_path)

    return {
        "name": name,
        "file": str(out_path),
        "source": str(source),
        "start": start,
        "end": end,
        "loop_len": loop_len,
        "y_scale": y_scale,
        "delta_y": base_delta_y,
        "repeats": LOOP_REPEATS,
        "frames": len(output_frames),
    }


def main() -> None:
    source = find_source()
    header, frames = split_rpl(source)
    if not frames:
        raise RuntimeError(f"No FRAME blocks found in {source}")

    print("Source:", source)
    print("Frames:", len(frames), "range", frames[0].frame, "->", frames[-1].frame)

    results = []
    for label, start, end, y_scale in VARIANTS:
        info = build_variant(source, header, frames, label, start, end, y_scale)
        results.append(info)
        print(f"made {info['name']} start={start} end={end} dy={info['delta_y']:.4f} scale={y_scale}")

    ref_path = OUT_DIR / "xioi_same_foot_loop_walk_v35_1_variants.json"
    ref_path.write_text(json.dumps({"version": "35.1", "variants": results}, indent=2), encoding="utf-8")
    print("\nReference:", ref_path)
    print("Copied to Steam replay root, parkour, and my replays.")
    print("If Toribash UI caches names, relaunch Toribash or refresh the replay list.")


if __name__ == "__main__":
    main()
