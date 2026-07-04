#!/usr/bin/env python3
"""
generate_xioi_auto_loop_finder_v36.py

V36 = auto-loop finder for Xioi walking RPL.

Goal:
  - start from the current good champion RPL
  - find natural loop points by comparing full body POS/QAT states
  - generate loop-walk replay candidates with corrected FIGHTNAME
  - copy them to Toribash replay/parkour and replay root

This does NOT use Lua/GRU. It keeps the full RPL physics context and only extends by
copying a segment whose start/end poses are similar.
"""
from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
STEAM_TORIBASH = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
REPLAY_ROOT = STEAM_TORIBASH / "replay"
REPLAY_PARKOUR = REPLAY_ROOT / "parkour"

SOURCE_CANDIDATES = [
    OUT_DIR / "xioi_master_final_v5_champion.rpl",
    OUT_DIR / "xioi_v30_23_mut.rpl",
    OUT_DIR / "xioi_source_template_v28.rpl",
]

# Search region. We keep the initial launch intact and look for a repeated walk pose after it.
SEARCH_START = 80
SEARCH_END = 360
MIN_LOOP = 45
MAX_LOOP = 130
TOP_N = 10
EXTRA_LOOPS = 3

# Body indices in Toribash POS arrays. These are deliberately broad.
# We weight core/shoulders/feet because the loop must not switch/lag foot contact.
CORE = [0, 1, 2, 3]
SHOULDERS = [11, 12]
HANDS = [9, 10]
HIPS_LEGS_FEET = [13, 14, 15, 16, 17, 18, 19, 20]
COMPARE_BODY_IDS = CORE + SHOULDERS + HIPS_LEGS_FEET


def pick_source() -> Path:
    for p in SOURCE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("No source champion found. Expected one of:\n" + "\n".join(map(str, SOURCE_CANDIDATES)))


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def parse_frames(lines: list[str]) -> dict[int, dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = {}
    current: int | None = None

    for line in lines:
        s = line.strip()
        m = re.match(r"FRAME\s+(\d+)\s*;", s)
        if m:
            current = int(m.group(1))
            frames.setdefault(current, {"lines": [], "pos": {}, "qat": {}, "joints": []})
            continue
        if current is None:
            continue
        frames[current]["lines"].append(line)

        # POS 0; x y z x y z ...
        m = re.match(r"POS\s+(\d+)\s*;\s*(.*)", s)
        if m:
            player = int(m.group(1))
            nums = [float(x) for x in m.group(2).split() if _is_float(x)]
            triples = [nums[i:i+3] for i in range(0, len(nums) - 2, 3)]
            frames[current]["pos"][player] = triples
            continue

        # QAT 0; w x y z ...
        m = re.match(r"QAT\s+(\d+)\s*;\s*(.*)", s)
        if m:
            player = int(m.group(1))
            nums = [float(x) for x in m.group(2).split() if _is_float(x)]
            quads = [nums[i:i+4] for i in range(0, len(nums) - 3, 4)]
            frames[current]["qat"][player] = quads
            continue

        m = re.match(r"JOINT\s+0\s*;\s*(.*)", s)
        if m:
            parts = m.group(1).split()
            pairs = []
            for i in range(0, len(parts)-1, 2):
                if parts[i].isdigit() and parts[i+1].lstrip('-').isdigit():
                    pairs.append((int(parts[i]), int(parts[i+1])))
            frames[current]["joints"].extend(pairs)

    return frames


def _is_float(x: str) -> bool:
    try:
        float(x)
        return True
    except ValueError:
        return False


def vec_dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(min(len(a), len(b)))))


def body_center(pos: list[list[float]], ids: list[int]) -> list[float] | None:
    pts = [pos[i] for i in ids if i < len(pos)]
    if not pts:
        return None
    return [sum(p[k] for p in pts) / len(pts) for k in range(3)]


def frame_score(frames: dict[int, dict[str, Any]], a: int, b: int) -> float | None:
    fa, fb = frames.get(a), frames.get(b)
    if not fa or not fb:
        return None
    pa = fa.get("pos", {}).get(0)
    pb = fb.get("pos", {}).get(0)
    if not pa or not pb:
        return None

    score = 0.0
    count = 0
    for idx in COMPARE_BODY_IDS:
        if idx < len(pa) and idx < len(pb):
            # Compare relative pose: remove whole-body translation in Y so forward progress is allowed.
            ca = body_center(pa, CORE) or [0, 0, 0]
            cb = body_center(pb, CORE) or [0, 0, 0]
            va = [pa[idx][0] - ca[0], pa[idx][1] - ca[1], pa[idx][2] - ca[2]]
            vb = [pb[idx][0] - cb[0], pb[idx][1] - cb[1], pb[idx][2] - cb[2]]
            w = 1.0
            if idx in SHOULDERS:
                w = 2.0
            if idx in [17, 18, 19, 20]:  # lower legs / feet-ish
                w = 2.5
            score += w * vec_dist(va, vb)
            count += 1
    if count == 0:
        return None

    # Penalize large vertical mismatch of core/shoulders.
    ca = body_center(pa, CORE) or [0, 0, 0]
    cb = body_center(pb, CORE) or [0, 0, 0]
    shoulder_a = body_center(pa, SHOULDERS) or ca
    shoulder_b = body_center(pb, SHOULDERS) or cb
    score += abs(ca[2] - cb[2]) * 4.0
    score += abs(shoulder_a[2] - shoulder_b[2]) * 4.0

    # Reward meaningful forward displacement, but not crazy huge jump.
    dy = cb[1] - ca[1]
    ady = abs(dy)
    if ady < 1.0:
        score += 50.0
    elif ady > 12.0:
        score += (ady - 12.0) * 8.0
    return score


def find_candidates(frames: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    keys = sorted(k for k in frames if SEARCH_START <= k <= SEARCH_END)
    candidates = []
    for a in keys:
        for b in keys:
            if b <= a:
                continue
            length = b - a
            if length < MIN_LOOP or length > MAX_LOOP:
                continue
            sc = frame_score(frames, a, b)
            if sc is None:
                continue
            pa = frames[a].get("pos", {}).get(0)
            pb = frames[b].get("pos", {}).get(0)
            ca = body_center(pa, CORE) if pa else None
            cb = body_center(pb, CORE) if pb else None
            dy = (cb[1] - ca[1]) if ca and cb else 0.0
            candidates.append({"start": a, "end": b, "length": length, "score": sc, "dy": dy})
    candidates.sort(key=lambda x: x["score"])
    return candidates[:TOP_N]


def set_fightname(lines: list[str], name: str) -> list[str]:
    out = []
    done = False
    for line in lines:
        if line.startswith("FIGHTNAME 0;"):
            out.append(f"FIGHTNAME 0; {name}")
            done = True
        else:
            out.append(line)
    if not done:
        out.insert(5, f"FIGHTNAME 0; {name}")
    return out


def split_by_frame(lines: list[str]) -> tuple[list[str], dict[int, list[str]]]:
    header = []
    blocks: dict[int, list[str]] = {}
    current = None
    for line in lines:
        m = re.match(r"FRAME\s+(\d+)\s*;", line.strip())
        if m:
            current = int(m.group(1))
            blocks[current] = [line]
        elif current is None:
            header.append(line)
        else:
            blocks[current].append(line)
    return header, blocks


def rewrite_frame_number(block: list[str], new_frame: int) -> list[str]:
    out = []
    first = True
    for line in block:
        if first and re.match(r"FRAME\s+\d+\s*;", line.strip()):
            out.append(f"FRAME {new_frame};")
            first = False
        else:
            out.append(line)
    return out


def make_loop_replay(source_lines: list[str], cand: dict[str, Any], variant_idx: int) -> tuple[str, list[str]]:
    start = int(cand["start"])
    end = int(cand["end"])
    length = int(cand["length"])
    name = f"xioi_auto_loop_v36_{variant_idx:02d}_s{start}_e{end}"

    header, blocks = split_by_frame(source_lines)
    header = set_fightname(header, name)

    original_keys = sorted(blocks)
    max_original = max(original_keys)
    out_lines = list(header)

    # Keep original through end of candidate segment.
    for fr in original_keys:
        if fr <= end:
            out_lines.extend(blocks[fr])

    segment_keys = [fr for fr in original_keys if start <= fr <= end]
    current_offset = end
    for loop_idx in range(EXTRA_LOOPS):
        base = current_offset + 5
        for fr in segment_keys:
            # Map start -> base, preserving relative frame spacing.
            new_fr = base + (fr - start)
            # Avoid duplicate exact frame if any.
            out_lines.extend(rewrite_frame_number(blocks[fr], new_fr))
        current_offset = base + (end - start)

    return name, out_lines


def write_outputs(source: Path, lines: list[str], candidates: list[dict[str, Any]]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
    REPLAY_PARKOUR.mkdir(parents=True, exist_ok=True)

    summary = {
        "version": 36,
        "source": str(source),
        "search_start": SEARCH_START,
        "search_end": SEARCH_END,
        "min_loop": MIN_LOOP,
        "max_loop": MAX_LOOP,
        "extra_loops": EXTRA_LOOPS,
        "candidates": candidates,
    }

    for i, cand in enumerate(candidates):
        name, rpl_lines = make_loop_replay(lines, cand, i)
        out = OUT_DIR / f"{name}.rpl"
        out.write_text("\n".join(rpl_lines) + "\n", encoding="utf-8")
        shutil.copy2(out, REPLAY_PARKOUR / out.name)
        shutil.copy2(out, REPLAY_ROOT / out.name)
        print(f"made {out.name} score={cand['score']:.3f} start={cand['start']} end={cand['end']} len={cand['length']} dy={cand['dy']:.3f}")

    ref = OUT_DIR / "xioi_auto_loop_v36_reference.json"
    ref.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Reference:", ref)
    print("Copied to:", REPLAY_PARKOUR)


def main() -> None:
    source = pick_source()
    lines = read_lines(source)
    frames = parse_frames(lines)
    print("Source:", source)
    print("Parsed frames:", len(frames))
    candidates = find_candidates(frames)
    if not candidates:
        raise RuntimeError("No loop candidates found. Try widening SEARCH_START/END or MIN/MAX_LOOP.")
    print("Top candidates:")
    for i, c in enumerate(candidates):
        print(f"  {i:02d}: score={c['score']:.3f} start={c['start']} end={c['end']} len={c['length']} dy={c['dy']:.3f}")
    write_outputs(source, lines, candidates)


if __name__ == "__main__":
    main()
