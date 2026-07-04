#!/usr/bin/env python3
"""
evolution_loop_xioi_gru_template_safe_v55.py

Human-in-the-loop evolution around the current GRU/template-safe walking champion.

Core idea:
- The launch is NOT learned/mutated here.
- Frames 0..315 are copied exactly from the champion RPL.
- Evolution only touches JOINT 0 commands after frame 315.
- POS/QAT/LINVEL/ANGVEL/FRAME structure is preserved, so the replay stays template-safe.

Usage:
  python3 scripts/evolution_loop_xioi_gru_template_safe_v55.py generate
  python3 scripts/evolution_loop_xioi_gru_template_safe_v55.py promote xioi_gru_v55_g001_c07.rpl
  python3 scripts/evolution_loop_xioi_gru_template_safe_v55.py status
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

ROOT = Path.home() / "Documents/ToribashAI"
OUT_DIR = ROOT / "generated_replays"
EVOL_DIR = OUT_DIR / "xioi_gru_v55_evolution"
STATE_PATH = EVOL_DIR / "xioi_gru_v55_state.json"
CHAMPION_PATH = OUT_DIR / "xioi_gru_v55_champion.rpl"

# Preferred parents, in order. First existing file is used if no champion exists yet.
PARENT_CANDIDATES = [
    OUT_DIR / "xioi_loop_len265_gru_v54_seed008.rpl",
    OUT_DIR / "xioi_loop_len265_gru_v54_seed048.rpl",
    OUT_DIR / "xioi_loop_len265_gru_v53_free_template_safe.rpl",
    OUT_DIR / "xioi_loop_len265_gru_v53_teacher_template_safe.rpl",
    OUT_DIR / "xioi_loop_len265_champion_v51.rpl",
]

STEAM_TORIBASH = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
STEAM_REPLAY = STEAM_TORIBASH / "replay"
STEAM_PARKOUR = STEAM_REPLAY / "parkour"

POP_SIZE = 16
PROTECTED_UNTIL = 315

# Keep the launch and first stable cycle extremely safe.
# Mutation gets gradually more permissive later.
def mutation_rate_for_frame(frame: int) -> float:
    if frame <= PROTECTED_UNTIL:
        return 0.0
    if frame <= 580:
        return 0.006   # very gentle: keep the first continuation stable
    if frame <= 950:
        return 0.014
    return 0.024

# Main balance/walk joints. Arms/shoulders/hips/legs are allowed after 315.
MUTABLE_JOINTS = [
    1, 3,          # chest/lumbar-ish stabilizers used by Xioi
    4, 5, 6, 7,   # arms/shoulders
    8, 9, 12, 13, # wrists/pec-ish support when present in source
    14, 15, 16, 17, 18, 19,  # hips/knees/ankles/feet
]

VALUE_CHOICES = [1, 2, 3, 4]

FRAME_RE = re.compile(r"^FRAME\s+(\d+)\s*;")
JOINT0_RE = re.compile(r"^JOINT\s+0\s*;\s*(.*)$")
FIGHTNAME_RE = re.compile(r"^FIGHTNAME\s+0\s*;")

@dataclass
class RPLData:
    lines: List[str]
    frame_for_line: Dict[int, int]


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"version": 55, "generation": 0, "champion": None, "history": []}


def save_state(state: dict) -> None:
    EVOL_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def pick_parent(state: dict) -> Path:
    if CHAMPION_PATH.exists():
        return CHAMPION_PATH
    champ = state.get("champion")
    if champ:
        p = Path(champ)
        if p.exists():
            return p
    for p in PARENT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "No parent replay found. Expected one of:\n"
        + "\n".join(str(p) for p in PARENT_CANDIDATES)
    )


def read_rpl(path: Path) -> RPLData:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    frame_for_line: Dict[int, int] = {}
    current_frame: Optional[int] = None
    for i, line in enumerate(lines):
        m = FRAME_RE.match(line)
        if m:
            current_frame = int(m.group(1))
        if current_frame is not None:
            frame_for_line[i] = current_frame
    return RPLData(lines=lines, frame_for_line=frame_for_line)


def parse_pairs(s: str) -> List[Tuple[int, int]]:
    nums = [int(x) for x in re.findall(r"-?\d+", s)]
    pairs = []
    for i in range(0, len(nums) - 1, 2):
        j, v = nums[i], nums[i + 1]
        if 0 <= j <= 19 and 1 <= v <= 4:
            pairs.append((j, v))
    return pairs


def format_pairs(pairs: List[Tuple[int, int]]) -> str:
    if not pairs:
        return "JOINT 0;"
    pairs = sorted(dict(pairs).items())
    tail = " ".join(f"{j} {v}" for j, v in pairs)
    return f"JOINT 0; {tail}"


def mutate_pairs(pairs: List[Tuple[int, int]], frame: int, rng: random.Random) -> Tuple[List[Tuple[int, int]], int]:
    rate = mutation_rate_for_frame(frame)
    if rate <= 0:
        return pairs, 0

    d = dict(pairs)
    mutations = 0

    # Mutate existing commands very gently.
    for j in list(d.keys()):
        if j not in MUTABLE_JOINTS:
            continue
        if rng.random() < rate:
            old = d[j]
            choices = [v for v in VALUE_CHOICES if v != old]
            # Prefer nearby state changes when possible.
            near = [v for v in choices if abs(v - old) == 1]
            d[j] = rng.choice(near or choices)
            mutations += 1

    # Rarely add a stabilizer command if the frame is sparse.
    if frame > 580 and len(d) < 7 and rng.random() < rate * 0.35:
        j = rng.choice(MUTABLE_JOINTS)
        if j not in d:
            d[j] = rng.choice(VALUE_CHOICES)
            mutations += 1

    # Very rarely remove a command late, never in the first continuation.
    if frame > 950 and len(d) > 2 and rng.random() < rate * 0.20:
        removable = [j for j in d.keys() if j in MUTABLE_JOINTS]
        if removable:
            del d[rng.choice(removable)]
            mutations += 1

    return sorted(d.items()), mutations


def set_fightname(lines: List[str], name: str) -> List[str]:
    out = []
    done = False
    for line in lines:
        if FIGHTNAME_RE.match(line):
            out.append(f"FIGHTNAME 0; {name}")
            done = True
        else:
            out.append(line)
    if not done:
        insert_at = 5 if len(out) > 5 else len(out)
        out.insert(insert_at, f"FIGHTNAME 0; {name}")
    return out


def mutate_rpl(parent: Path, out_path: Path, seed: int, candidate_index: int, generation: int) -> dict:
    rng = random.Random(seed)
    rpl = read_rpl(parent)
    out_lines = list(rpl.lines)
    total_mutations = 0
    touched_frames = 0

    for i, line in enumerate(out_lines):
        m = JOINT0_RE.match(line)
        if not m:
            continue
        frame = rpl.frame_for_line.get(i)
        if frame is None or frame <= PROTECTED_UNTIL:
            continue
        pairs = parse_pairs(m.group(1))
        new_pairs, n = mutate_pairs(pairs, frame, rng)
        if n:
            out_lines[i] = format_pairs(new_pairs)
            total_mutations += n
            touched_frames += 1

    name = out_path.stem
    out_lines = set_fightname(out_lines, name)
    out_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    return {
        "candidate": candidate_index,
        "generation": generation,
        "seed": seed,
        "parent": str(parent),
        "rpl": str(out_path),
        "protected_until": PROTECTED_UNTIL,
        "mutations": total_mutations,
        "touched_frames": touched_frames,
    }


def copy_to_steam(path: Path) -> None:
    STEAM_REPLAY.mkdir(parents=True, exist_ok=True)
    STEAM_PARKOUR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, STEAM_REPLAY / path.name)
    shutil.copy2(path, STEAM_PARKOUR / path.name)


def generate(args: argparse.Namespace) -> None:
    state = load_state()
    parent = pick_parent(state)
    gen = int(state.get("generation", 0)) + 1
    pop = int(args.population or POP_SIZE)

    EVOL_DIR.mkdir(parents=True, exist_ok=True)
    gen_dir = EVOL_DIR / f"g{gen:03d}"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Also copy parent for visual comparison.
    parent_out = gen_dir / f"xioi_gru_v55_g{gen:03d}_c00_PARENT.rpl"
    shutil.copy2(parent, parent_out)
    parent_lines = set_fightname(parent_out.read_text(encoding="utf-8", errors="ignore").splitlines(), parent_out.stem)
    parent_out.write_text("\n".join(parent_lines) + "\n", encoding="utf-8")
    copy_to_steam(parent_out)

    summaries = []
    base_seed = args.seed if args.seed is not None else random.randrange(1_000_000_000)
    for c in range(1, pop + 1):
        out = gen_dir / f"xioi_gru_v55_g{gen:03d}_c{c:02d}.rpl"
        summary = mutate_rpl(parent, out, seed=base_seed + c * 1009, candidate_index=c, generation=gen)
        copy_to_steam(out)
        summaries.append(summary)

    summary_path = gen_dir / f"xioi_gru_v55_g{gen:03d}_summary.json"
    summary_path.write_text(json.dumps({
        "version": 55,
        "generation": gen,
        "parent": str(parent),
        "population": pop,
        "protected_until": PROTECTED_UNTIL,
        "base_seed": base_seed,
        "candidates": summaries,
    }, indent=2), encoding="utf-8")

    state["generation"] = gen
    state["last_generation_dir"] = str(gen_dir)
    save_state(state)

    print(f"Generation {gen} created from parent:")
    print(parent)
    print(f"Candidates: {gen_dir}")
    print("Copied to Steam replay root + parkour.")
    print("In Toribash UI, look for xioi_gru_v55_g%03d_cXX" % gen)


def find_replay_by_name(name: str) -> Path:
    p = Path(name)
    if p.exists():
        return p
    state = load_state()
    search_dirs = [Path(state.get("last_generation_dir", EVOL_DIR)), EVOL_DIR, OUT_DIR, STEAM_REPLAY, STEAM_PARKOUR]
    for d in search_dirs:
        if not d.exists():
            continue
        matches = list(d.rglob(name)) if not name.endswith(".rpl") else list(d.rglob(name))
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Could not find replay: {name}")


def promote(args: argparse.Namespace) -> None:
    src = find_replay_by_name(args.replay)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, CHAMPION_PATH)
    lines = set_fightname(CHAMPION_PATH.read_text(encoding="utf-8", errors="ignore").splitlines(), CHAMPION_PATH.stem)
    CHAMPION_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    copy_to_steam(CHAMPION_PATH)

    state = load_state()
    state["champion"] = str(CHAMPION_PATH)
    state.setdefault("history", []).append({"promoted": str(src), "champion": str(CHAMPION_PATH)})
    save_state(state)

    print("Promoted:", src)
    print("Champion:", CHAMPION_PATH)
    print("Next generate will mutate from this champion.")


def status(_: argparse.Namespace) -> None:
    state = load_state()
    print(json.dumps(state, indent=2))
    if CHAMPION_PATH.exists():
        print("Champion exists:", CHAMPION_PATH)
    else:
        print("No V55 champion yet. First parent will be:", pick_parent(state))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("--population", type=int, default=POP_SIZE)
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.set_defaults(func=generate)

    p_promote = sub.add_parser("promote")
    p_promote.add_argument("replay")
    p_promote.set_defaults(func=promote)

    p_status = sub.add_parser("status")
    p_status.set_defaults(func=status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
