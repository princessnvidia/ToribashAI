#!/usr/bin/env python3
"""
generate_xioi_template_mutations_v28.py

Crée une petite population de replays candidats à partir du replay source Xioi.

Important:
  Cette version garde le contexte source autant que possible et ne modifie que
  légèrement les JOINT dans les premières frames. C'est fait pour exploration
  visuelle / sélection manuelle, pas encore pour scoring automatique.

Sortie:
  generated_replays/xioi_v28_candidates/*.rpl
  copie dans Toribash/replay/
"""

from __future__ import annotations

import json
import random
import re
import shutil
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
TEMPLATE = OUT_DIR / "xioi_source_template_v28.rpl"
CAND_DIR = OUT_DIR / "xioi_v28_candidates"
STEAM_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)
SUMMARY = OUT_DIR / "xioi_v28_candidates_summary.json"

RANDOM_SEED = 2801
CANDIDATES = 16
MUTATE_UNTIL_FRAME = 260
MUTATION_RATE = 0.08
KEEP_RATE = 0.92
JOINT_VALUES = [1, 2, 3, 4]

# On évite de brutaliser tout le corps. On privilégie jambes / hanches / bras utiles.
MUTABLE_JOINTS = set(range(20))

FRAME_RE = re.compile(r"^FRAME\s+(\d+)\s*;")
JOINT_RE = re.compile(r"^(JOINT\s+0\s*;\s*)(\d+)\s+([1-4])\s*$")


def current_frame_from_line(line: str, current: int) -> int:
    m = FRAME_RE.match(line.strip())
    if m:
        return int(m.group(1))
    return current


def mutate_lines(lines: list[str], rng: random.Random, strength: float) -> tuple[list[str], int]:
    out: list[str] = []
    frame = -1
    mutations = 0

    for line in lines:
        frame = current_frame_from_line(line, frame)
        m = JOINT_RE.match(line.strip())
        if m and 0 <= frame <= MUTATE_UNTIL_FRAME:
            prefix, j_s, v_s = m.groups()
            j = int(j_s)
            v = int(v_s)
            if j in MUTABLE_JOINTS and rng.random() < MUTATION_RATE * strength:
                choices = [x for x in JOINT_VALUES if x != v]
                new_v = rng.choice(choices)
                out.append(f"{prefix}{j} {new_v}\n")
                mutations += 1
                continue
        out.append(line)
    return out, mutations


def rename_fight(lines: list[str], name: str) -> list[str]:
    out = []
    for line in lines:
        if line.startswith("FIGHTNAME"):
            out.append(f"FIGHTNAME 0; {name}\n")
        elif line.startswith("AUTHOR"):
            out.append("AUTHOR 0; ToribashAI V28\n")
        else:
            out.append(line)
    return out


def main() -> None:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Template absent: {TEMPLATE}. Lance d'abord build_xioi_template_v28.py")

    CAND_DIR.mkdir(parents=True, exist_ok=True)
    STEAM_REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(RANDOM_SEED)

    base_lines = TEMPLATE.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    made = []

    # candidat 0 = exact source
    exact_name = "xioi_v28_00_exact_source"
    exact_path = CAND_DIR / f"{exact_name}.rpl"
    exact_path.write_text("".join(rename_fight(base_lines, exact_name)), encoding="utf-8")
    shutil.copy2(exact_path, STEAM_REPLAY_DIR / exact_path.name)
    made.append({"file": str(exact_path), "mutations": 0, "type": "exact"})

    for i in range(1, CANDIDATES + 1):
        rng = random.Random(RANDOM_SEED + i)
        strength = 0.5 + i / CANDIDATES
        lines, mutations = mutate_lines(base_lines, rng, strength)
        name = f"xioi_v28_{i:02d}_mut{mutations:03d}"
        lines = rename_fight(lines, name)
        out_path = CAND_DIR / f"{name}.rpl"
        out_path.write_text("".join(lines), encoding="utf-8")
        shutil.copy2(out_path, STEAM_REPLAY_DIR / out_path.name)
        made.append({"file": str(out_path), "mutations": mutations, "strength": strength})
        print("made:", out_path.name, "mutations:", mutations)

    SUMMARY.write_text(json.dumps({"version": 28, "template": str(TEMPLATE), "candidates": made}, indent=2), encoding="utf-8")
    print("\nCopied to Steam replay dir:", STEAM_REPLAY_DIR)
    print("Summary:", SUMMARY)


if __name__ == "__main__":
    main()
