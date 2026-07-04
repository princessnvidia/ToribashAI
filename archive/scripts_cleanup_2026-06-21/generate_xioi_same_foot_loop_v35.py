#!/usr/bin/env python3
"""
generate_xioi_same_foot_loop_v35.py

V35 = loop-walk reference generator.
Goal:
  - start from the current good Xioi champion RPL
  - keep the launch / first walk intact
  - extend walking by repeating ONE coherent same-foot cycle
  - avoid the bad alternation bug where the loop swaps right/left then goes backward
  - use the usual ToribashAI flat/goal mod if present, otherwise keep source mod

Output:
  generated_replays/xioi_same_foot_loop_walk_v35.rpl
  generated_replays/xioi_same_foot_loop_walk_v35_reference.json
  copied to Steam replay/parkour/

This script is intentionally conservative: it does not invent new poses, it copies a later
cycle from the champion and translates POS-like lines when present so the body continues
at the same displacement step instead of snapping backward.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Iterable

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
CHAMPION_CANDIDATES = [
    OUT_DIR / "xioi_master_final_v5_champion.rpl",
    OUT_DIR / "xioi_v30_23_mut.rpl",
    OUT_DIR / "xioi_master_final_v5_loop_10.rpl",
]
STEAM_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)
STEAM_PARKOUR_DIR = STEAM_REPLAY_DIR / "parkour"
OUT_RPL = OUT_DIR / "xioi_same_foot_loop_walk_v35.rpl"
OUT_REF = OUT_DIR / "xioi_same_foot_loop_walk_v35_reference.json"

# Conservative defaults. If the champion has a longer clean walk, adjust these.
# We preserve the launch and first successful steps, then repeat a late cycle.
PROTECT_UNTIL = 150
LOOP_START = 150
LOOP_END = 230
REPEATS = 5

USUAL_MOD = "ToribashAI/toribashai_xioi_city_v1.tbm"
FALLBACK_MOD = "classic"

FRAME_RE = re.compile(r"^FRAME\s+(-?\d+)\s*;")
FIGHTNAME_RE = re.compile(r"^FIGHTNAME\s+0\s*;")
JOINT_RE = re.compile(r"^JOINT\s+(\d+)\s*;\s*(.*)$")
POS_RE = re.compile(r"^(POS|LINVEL|ANGVEL|QAT)\s+(\d+)\s*;\s*(.*)$")
ENGAGE_RE = re.compile(r"^ENGAGE\s+(\d+)\s*;\s*(.*)$")
NEWGAME_RE = re.compile(r"^NEWGAME\s+0\s*;(.*)$")


def find_champion() -> Path:
    for p in CHAMPION_CANDIDATES:
        if p.exists():
            return p
    candidates = sorted(OUT_DIR.glob("*champion*.rpl")) + sorted(OUT_DIR.glob("xioi_v30_23_mut.rpl"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError("No Xioi champion RPL found in generated_replays")


def split_blocks(lines: list[str]) -> tuple[list[str], list[tuple[int, list[str]]]]:
    header: list[str] = []
    blocks: list[tuple[int, list[str]]] = []
    current_frame: int | None = None
    current: list[str] = []

    for line in lines:
        m = FRAME_RE.match(line)
        if m:
            if current_frame is not None:
                blocks.append((current_frame, current))
            else:
                header = current
            current_frame = int(m.group(1))
            current = [line]
        else:
            current.append(line)

    if current_frame is not None:
        blocks.append((current_frame, current))
    else:
        header = current
    return header, blocks


def block_frame(block: list[str], new_frame: int) -> list[str]:
    out = []
    for line in block:
        if FRAME_RE.match(line):
            out.append(f"FRAME {new_frame};")
        else:
            out.append(line)
    return out


def parse_first_pos_y(blocks: Iterable[tuple[int, list[str]]]) -> dict[int, float]:
    """Very rough body Y estimate from first POS line per frame if present."""
    y_by_frame: dict[int, float] = {}
    for fr, block in blocks:
        for line in block:
            if line.startswith("POS 0;"):
                vals = [float(x) for x in line.split(";", 1)[1].strip().split()]
                # Toribash POS line is flat xyz triples. Average Y of first 21 body parts.
                ys = vals[1::3]
                if ys:
                    y_by_frame[fr] = sum(ys) / len(ys)
                break
    return y_by_frame


def infer_loop_delta_y(blocks: list[tuple[int, list[str]]], start: int, end: int) -> float:
    y = parse_first_pos_y(blocks)
    if not y:
        return 0.0
    start_keys = [k for k in y if abs(k - start) <= 20]
    end_keys = [k for k in y if abs(k - end) <= 20]
    if not start_keys or not end_keys:
        return 0.0
    s = min(start_keys, key=lambda k: abs(k - start))
    e = min(end_keys, key=lambda k: abs(k - end))
    return y[e] - y[s]


def translate_numeric_line(line: str, dy: float) -> str:
    """Translate POS-like Y values by dy. Keep QAT unchanged."""
    m = POS_RE.match(line)
    if not m:
        return line
    kind, player, rest = m.groups()
    if kind != "POS":
        return line
    try:
        vals = [float(x) for x in rest.strip().split()]
    except ValueError:
        return line
    if len(vals) < 3:
        return line
    for i in range(1, len(vals), 3):
        vals[i] += dy
    formatted = " ".join(f"{v:.8f}" for v in vals)
    return f"{kind} {player}; {formatted}"


def translate_block(block: list[str], dy: float) -> list[str]:
    return [translate_numeric_line(line, dy) for line in block]


def set_fightname(header: list[str], name: str) -> list[str]:
    out = []
    done = False
    for line in header:
        if FIGHTNAME_RE.match(line):
            out.append(f"FIGHTNAME 0; {name}")
            done = True
        elif NEWGAME_RE.match(line):
            # Use usual mod if available in Toribash data/mod, otherwise classic.
            mod_path = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/mod" / USUAL_MOD
            chosen_mod = USUAL_MOD if mod_path.exists() else FALLBACK_MOD
            # Replace only final token if it looks like a mod token.
            prefix = line.rsplit(" ", 1)[0]
            out.append(prefix + " " + chosen_mod)
        else:
            out.append(line)
    if not done:
        insert_at = 5 if len(out) > 5 else len(out)
        out.insert(insert_at, f"FIGHTNAME 0; {name}")
    return out


def collect_reference(blocks: list[tuple[int, list[str]]]) -> dict:
    frames = []
    for fr, block in blocks:
        joints = []
        has_pos = False
        for line in block:
            jm = JOINT_RE.match(line)
            if jm:
                player = int(jm.group(1))
                parts = jm.group(2).strip().split()
                pairs = []
                for i in range(0, len(parts) - 1, 2):
                    try:
                        pairs.append([int(parts[i]), int(parts[i + 1])])
                    except ValueError:
                        pass
                if pairs:
                    joints.append({"player": player, "pairs": pairs})
            if line.startswith("POS 0;"):
                has_pos = True
        frames.append({"frame": fr, "joints": joints, "has_pos": has_pos})
    return {
        "name": "xioi_same_foot_loop_walk_v35_reference",
        "source": str(OUT_RPL),
        "protect_until": PROTECT_UNTIL,
        "loop_start": LOOP_START,
        "loop_end": LOOP_END,
        "repeats": REPEATS,
        "frames": frames,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STEAM_PARKOUR_DIR.mkdir(parents=True, exist_ok=True)

    src = find_champion()
    print("Source champion:", src)
    lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
    header, blocks = split_blocks(lines)
    if not blocks:
        raise RuntimeError("No FRAME blocks found")

    blocks = sorted(blocks, key=lambda x: x[0])
    initial = [(fr, b) for fr, b in blocks if fr < LOOP_END]
    loop = [(fr, b) for fr, b in blocks if LOOP_START <= fr < LOOP_END]
    tail = [(fr, b) for fr, b in blocks if fr >= LOOP_END]
    if not loop:
        raise RuntimeError(f"No loop blocks found in {LOOP_START}-{LOOP_END}")

    loop_len = LOOP_END - LOOP_START
    delta_y = infer_loop_delta_y(blocks, LOOP_START, LOOP_END)
    print("Loop frames:", LOOP_START, "->", LOOP_END, "len", loop_len)
    print("Inferred loop delta Y:", round(delta_y, 4))
    if abs(delta_y) < 0.01:
        print("Warning: no POS delta inferred; copying JOINT timing only.")

    new_blocks: list[tuple[int, list[str]]] = []
    # Keep the learned launch and first steps untouched until LOOP_END.
    for fr, block in initial:
        new_blocks.append((fr, block))

    # Repeat same cycle at same relative distance. No left/right swapping.
    for r in range(1, REPEATS + 1):
        frame_offset = r * loop_len
        y_offset = r * delta_y
        for fr, block in loop:
            new_frame = fr + frame_offset
            nb = block_frame(block, new_frame)
            nb = translate_block(nb, y_offset)
            nb.insert(1, f"# v35 same-foot loop repeat={r} source_frame={fr}")
            new_blocks.append((new_frame, nb))

    # Optional: append tail translated after final loop if it exists, but do not let it snap back.
    final_frame_offset = REPEATS * loop_len
    final_y_offset = REPEATS * delta_y
    for fr, block in tail[:20]:
        new_frame = fr + final_frame_offset
        nb = block_frame(block, new_frame)
        nb = translate_block(nb, final_y_offset)
        nb.insert(1, f"# v35 translated tail source_frame={fr}")
        new_blocks.append((new_frame, nb))

    new_blocks = sorted(new_blocks, key=lambda x: x[0])
    name = OUT_RPL.stem
    header = set_fightname(header, name)

    out_lines: list[str] = []
    out_lines.extend(header)
    if out_lines and out_lines[-1].strip():
        out_lines.append("")
    for _, block in new_blocks:
        out_lines.extend(block)
        out_lines.append("")

    OUT_RPL.write_text("\n".join(out_lines), encoding="utf-8")
    OUT_REF.write_text(json.dumps(collect_reference(new_blocks), indent=2), encoding="utf-8")
    shutil.copy2(OUT_RPL, STEAM_PARKOUR_DIR / OUT_RPL.name)
    shutil.copy2(OUT_RPL, STEAM_REPLAY_DIR / OUT_RPL.name)
    print("Wrote:", OUT_RPL)
    print("Reference:", OUT_REF)
    print("Copied to:", STEAM_PARKOUR_DIR / OUT_RPL.name)
    print("Also copied to replay root for compatibility.")


if __name__ == "__main__":
    main()
