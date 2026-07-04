#!/usr/bin/env python3
"""
generate_xioi_stable_loop_dataset_v49.py

V49 = extract stable walking loops from the best V48 replay and build:
  - a long pure-loop RPL made only from the visually stable loop region
  - a reference JSON
  - an imitation dataset JSONL for future GRU training

User-selected stable-ish loops from V48:
  loop 1: 485 -> 750
  loop 2: 750 -> 1010
  loop 3: 1010 -> 1270

Because all three have a slight delay, V49 generates variants with small timing offsets:
  base, early10, early20, early30

The default champion/reference candidate is early20, usually a good first correction for "slight delay".
"""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
DATASET_DIR = ROOT / "datasets" / "ml"
STEAM_TORIBASH = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
STEAM_REPLAY_ROOT = STEAM_TORIBASH / "replay"
STEAM_REPLAY_PARKOUR = STEAM_REPLAY_ROOT / "parkour"

# Prefer V48 if present. Fall back to the closest V47 phase150, then V46.
SOURCE_CANDIDATES = [
    OUT_DIR / "xioi_assassin_template_loop_v48.rpl",
    OUT_DIR / "xioi_assassin_template_loop_v47_phase150.rpl",
    OUT_DIR / "xioi_assassin_template_loop_v46.rpl",
]

BASE_NAME = "xioi_stable_loop_v49"
SEQ_LEN = 8
ACTION_DIM = 20
REPEAT_CYCLES = 12

# User selected frames. We will derive the stable cycle by concatenating these regions.
LOOPS = [(485, 750), (750, 1010), (1010, 1270)]
# Negative offset = start earlier and end earlier to compensate "slight delay".
VARIANTS = {
    "base": 0,
    "early10": -10,
    "early20": -20,
    "early30": -30,
}
DEFAULT_VARIANT = "early20"

FRAME_RE = re.compile(r"^FRAME\s+(-?\d+)\s*;")
JOINT_RE = re.compile(r"^JOINT\s+(\d+)\s*;\s*(.*)$")
FIGHTNAME_RE = re.compile(r"^FIGHTNAME\s+0\s*;")
NEWGAME_RE = re.compile(r"^NEWGAME\s+0\s*;")


def find_source() -> Path:
    for p in SOURCE_CANDIDATES:
        if p.exists():
            return p
    # Last chance: any generated V48-like replay.
    matches = sorted(OUT_DIR.glob("*v48*.rpl"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Missing source V48/V47/V46 replay. Expected one of:\n"
        + "\n".join(str(p) for p in SOURCE_CANDIDATES)
    )


def parse_rpl_blocks(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    header: list[str] = []
    blocks: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

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

    blocks.sort(key=lambda b: int(b["frame"]))
    return header, blocks


def parse_joint_pairs_from_lines(lines: list[str], player: int = 0) -> dict[int, int]:
    pairs: dict[int, int] = {}
    for line in lines:
        m = JOINT_RE.match(line.strip())
        if not m:
            continue
        if int(m.group(1)) != player:
            continue
        nums = [int(x) for x in re.findall(r"-?\d+", m.group(2))]
        for i in range(0, len(nums) - 1, 2):
            j, v = nums[i], nums[i + 1]
            if 0 <= j < ACTION_DIM and 0 <= v <= 4:
                pairs[j] = v
    return pairs


def line_without_joint0(lines: list[str]) -> list[str]:
    out = []
    for line in lines:
        m = JOINT_RE.match(line.strip())
        if m and int(m.group(1)) == 0:
            continue
        out.append(line)
    return out


def compact_joint_line(pairs: dict[int, int]) -> str | None:
    if not pairs:
        return None
    flat = []
    for j in sorted(pairs):
        flat.extend([str(j), str(pairs[j])])
    return "JOINT 0; " + " ".join(flat)


def build_action_map(blocks: list[dict[str, Any]]) -> dict[int, dict[int, int]]:
    return {int(b["frame"]): parse_joint_pairs_from_lines(b["lines"], player=0) for b in blocks}


def choose_frames_for_variant(offset: int, action_map: dict[int, dict[int, int]]) -> list[int]:
    source_frames = sorted(action_map)
    selected: list[int] = []
    for a, b in LOOPS:
        aa, bb = a + offset, b + offset
        selected.extend([f for f in source_frames if aa <= f < bb])
    # Deduplicate while preserving order.
    seen = set()
    clean = []
    for f in selected:
        if f not in seen:
            seen.add(f)
            clean.append(f)
    return clean


def patch_header(header: list[str], fightname: str, matchframes: int) -> list[str]:
    out = []
    has_fight = False
    for line in header:
        if FIGHTNAME_RE.match(line):
            out.append(f"FIGHTNAME 0; {fightname}")
            has_fight = True
        elif NEWGAME_RE.match(line):
            # Keep all game settings/mod after matchframes when possible.
            prefix, rest = line.split(";", 1)
            parts = rest.strip().split()
            if parts:
                parts[0] = str(matchframes)
                out.append(prefix + ";" + " ".join(parts))
            else:
                out.append(line)
        else:
            out.append(line)
    if not has_fight:
        insert_at = 5 if len(out) > 5 else len(out)
        out.insert(insert_at, f"FIGHTNAME 0; {fightname}")
    return out


def make_rpl_variant(source: Path, header: list[str], blocks: list[dict[str, Any]], action_map: dict[int, dict[int, int]], variant: str, offset: int) -> tuple[Path, dict[str, Any]]:
    chosen_frames = choose_frames_for_variant(offset, action_map)
    if not chosen_frames:
        raise RuntimeError(f"No frames selected for variant {variant}")

    # Use a representative template block for POS/QAT structure. We preserve copied block lines but renumber frames.
    block_by_frame = {int(b["frame"]): b for b in blocks}
    cycle_len = len(chosen_frames)
    generated_blocks: list[list[str]] = []
    new_frame = 0
    frame_step_values: list[int] = []

    # Estimate frame step from selected frames.
    diffs = [b - a for a, b in zip(chosen_frames, chosen_frames[1:]) if b > a]
    frame_step = min(diffs) if diffs else 5
    if frame_step <= 0 or frame_step > 20:
        frame_step = 5

    all_actions: list[dict[str, Any]] = []
    for rep in range(REPEAT_CYCLES):
        for src_frame in chosen_frames:
            src_block = block_by_frame[src_frame]
            base_lines = line_without_joint0(src_block["lines"])
            # Replace FRAME line with sequential loop timeline.
            rewritten = []
            for i, line in enumerate(base_lines):
                if i == 0 and FRAME_RE.match(line):
                    rewritten.append(f"FRAME {new_frame}; 0 0 0 0")
                else:
                    rewritten.append(line)
            pairs = action_map.get(src_frame, {})
            joint_line = compact_joint_line(pairs)
            if joint_line:
                # Insert right after FRAME line.
                rewritten.insert(1, joint_line)
            generated_blocks.append(rewritten)
            all_actions.append({
                "frame": new_frame,
                "source_frame": src_frame,
                "pairs": [[j, v] for j, v in sorted(pairs.items())],
                "active": len(pairs),
                "cycle_repeat": rep,
            })
            frame_step_values.append(frame_step)
            new_frame += frame_step

    matchframes = max(1000, new_frame + 100)
    fightname = f"{BASE_NAME}_{variant}"
    out_lines = patch_header(header, fightname, matchframes)
    out_lines.append("")
    for bl in generated_blocks:
        out_lines.extend(bl)
        out_lines.append("")

    out_path = OUT_DIR / f"{fightname}.rpl"
    out_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")

    # Copy to Steam replay dirs.
    STEAM_REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
    STEAM_REPLAY_PARKOUR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, STEAM_REPLAY_ROOT / out_path.name)
    shutil.copy2(out_path, STEAM_REPLAY_PARKOUR / out_path.name)

    counts = Counter()
    active = Counter()
    for a in all_actions:
        active[a["active"]] += 1
        vals = [0] * ACTION_DIM
        for j, v in a["pairs"]:
            vals[j] = v
        counts.update(vals)

    meta = {
        "version": 49,
        "variant": variant,
        "offset": offset,
        "source_rpl": str(source),
        "output_rpl": str(out_path),
        "chosen_source_frames_min": min(chosen_frames),
        "chosen_source_frames_max": max(chosen_frames),
        "chosen_source_frame_count": len(chosen_frames),
        "repeat_cycles": REPEAT_CYCLES,
        "generated_actions": len(all_actions),
        "matchframes": matchframes,
        "value_counts": counts.most_common(),
        "active_distribution": active.most_common(),
        "actions": all_actions,
    }
    return out_path, meta


def build_reference_and_dataset(default_meta: dict[str, Any]) -> None:
    ref_path = OUT_DIR / f"{BASE_NAME}_reference.json"
    ref = {k: v for k, v in default_meta.items() if k != "actions"}
    ref["actions"] = default_meta["actions"]
    ref_path.write_text(json.dumps(ref, indent=2), encoding="utf-8")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = DATASET_DIR / f"{BASE_NAME}_sequences.jsonl"
    actions = default_meta["actions"]
    rows = 0
    with dataset_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(actions) - SEQ_LEN):
            seq = []
            for a in actions[i:i + SEQ_LEN]:
                vals = [0] * ACTION_DIM
                for j, v in a["pairs"]:
                    vals[j] = v
                seq.append(vals)
            target = [0] * ACTION_DIM
            for j, v in actions[i + SEQ_LEN]["pairs"]:
                target[j] = v
            row = {
                "seq": seq,
                "target": target,
                "frame": actions[i + SEQ_LEN]["frame"],
                "source_frame": actions[i + SEQ_LEN]["source_frame"],
                "variant": default_meta["variant"],
            }
            f.write(json.dumps(row) + "\n")
            rows += 1

    summary = {
        "version": 49,
        "reference": str(ref_path),
        "dataset": str(dataset_path),
        "rows": rows,
        "seq_len": SEQ_LEN,
        "action_dim": ACTION_DIM,
        "source_variant": default_meta["variant"],
        "chosen_source_frames_min": default_meta["chosen_source_frames_min"],
        "chosen_source_frames_max": default_meta["chosen_source_frames_max"],
        "chosen_source_frame_count": default_meta["chosen_source_frame_count"],
        "value_counts": default_meta["value_counts"],
        "active_distribution": default_meta["active_distribution"],
    }
    (OUT_DIR / f"{BASE_NAME}_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Reference:", ref_path)
    print("Dataset:", dataset_path)
    print("Dataset summary:")
    print(json.dumps(summary, indent=2))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source = find_source()
    print("Source:", source)
    header, blocks = parse_rpl_blocks(source)
    action_map = build_action_map(blocks)
    print("Frames:", len(blocks), "Action frames:", sum(1 for v in action_map.values() if v))
    print("User loops:", LOOPS)

    variant_metas: dict[str, dict[str, Any]] = {}
    for variant, offset in VARIANTS.items():
        out_path, meta = make_rpl_variant(source, header, blocks, action_map, variant, offset)
        variant_metas[variant] = meta
        print(f"Made {variant}: {out_path.name} frames={meta['generated_actions']} src={meta['chosen_source_frames_min']}..{meta['chosen_source_frames_max']}")

    # Also create a canonical copy from default variant.
    champion_src = OUT_DIR / f"{BASE_NAME}_{DEFAULT_VARIANT}.rpl"
    champion = OUT_DIR / f"{BASE_NAME}_champion_candidate.rpl"
    shutil.copy2(champion_src, champion)
    shutil.copy2(champion, STEAM_REPLAY_ROOT / champion.name)
    shutil.copy2(champion, STEAM_REPLAY_PARKOUR / champion.name)
    print("Champion candidate:", champion)

    build_reference_and_dataset(variant_metas[DEFAULT_VARIANT])


if __name__ == "__main__":
    main()
