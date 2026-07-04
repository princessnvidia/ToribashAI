#!/usr/bin/env python3
"""
generate_xioi_template_mutations_v30.py

V30 = génération suivante autour du champion V29.

Principe :
- On part du meilleur V29, idéalement xioi_v29_champion.rpl.
- Si le champion officiel n'existe pas encore, on essaie xioi_v29_23_mut.rpl.
- On garde tout le replay source : POS/QAT/NEWGAME/ENGAGE/contexte physique.
- On mute seulement quelques lignes JOINT, avec un gros biais sur les premières frames.

Sorties :
- generated_replays/xioi_v30_parent.rpl
- generated_replays/xioi_v30_01_mut.rpl ... xioi_v30_32_mut.rpl
- copie automatique dans le dossier replay Toribash.
"""

from __future__ import annotations

import random
import re
import shutil
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
TORIBASH_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)

# Parent préféré : champion promu. Fallback : ta sélection V29 mutation 23.
PARENT_CANDIDATES = [
    OUT_DIR / "xioi_v29_champion.rpl",
    OUT_DIR / "xioi_v29_23_mut.rpl",
    OUT_DIR / "xioi_v29_parent.rpl",
    OUT_DIR / "xioi_source_template_v28.rpl",
]

GENERATION = 30
NUM_MUTATIONS = 32
RANDOM_SEED = 30023

# Mutations plus ciblées que V29 : on garde la marche jolie, on cherche de petites améliorations.
BASE_MUTATION_RATE = 0.055
EARLY_MUTATION_RATE = 0.145
MID_MUTATION_RATE = 0.075
LATE_MUTATION_RATE = 0.025

EARLY_END_FRAME = 120
MID_END_FRAME = 260

# Les joints changés trop brutalement cassent vite la mécanique.
MAX_TOTAL_MUTATIONS_MIN = 4
MAX_TOTAL_MUTATIONS_MAX = 14

# Probabilité de garder une mutation même si elle touche un bras.
# Bas, parce que les bras stabilisent beaucoup Xioi.
ARM_MUTATION_KEEP = 0.38

# Joints Toribash usuels :
# 0 neck, 1 chest, 2 lumbar, 3 abs, 4/5 pecs, 6/7 shoulders, 8/9 elbows,
# 10/11 wrists, 12/13 glutes, 14/15 hips, 16/17 knees, 18/19 ankles.
EARLY_FOCUS_JOINTS = {1, 2, 3, 12, 13, 14, 15, 16, 17, 18, 19}
ARMS = {4, 5, 6, 7, 8, 9, 10, 11}

JOINT_RE = re.compile(r"^(JOINT\s+0;\s+)(\d+)\s+(\d+)(.*)$")
FRAME_RE = re.compile(r"^FRAME\s+(\d+);")
FIGHTNAME_RE = re.compile(r"^FIGHTNAME\s+0;.*$")
AUTHOR_RE = re.compile(r"^AUTHOR\s+0;.*$")


def find_parent() -> Path:
    for p in PARENT_CANDIDATES:
        if p.exists():
            return p
    matches = sorted(OUT_DIR.glob("*v29*23*.rpl"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Aucun parent V29 trouvé. Lance d'abord :\n"
        "python3 scripts/promote_xioi_v29_best.py xioi_v29_23_mut.rpl"
    )


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def mutation_rate_for(frame: int, joint: int) -> float:
    if frame <= EARLY_END_FRAME:
        rate = EARLY_MUTATION_RATE
    elif frame <= MID_END_FRAME:
        rate = MID_MUTATION_RATE
    else:
        rate = LATE_MUTATION_RATE

    if joint in EARLY_FOCUS_JOINTS:
        rate *= 1.25
    elif joint in ARMS:
        rate *= ARM_MUTATION_KEEP
    else:
        rate *= 0.75

    return min(rate, 0.30)


def mutate_value(value: int) -> int:
    # Valeurs Toribash joints : 1..4 généralement. On évite 0 dans les RPL humains.
    choices = [1, 2, 3, 4]
    if value in choices:
        # Petit déplacement local d'abord, parfois saut vers autre état.
        if random.random() < 0.78:
            candidates = [v for v in (value - 1, value + 1) if 1 <= v <= 4]
            if candidates:
                return random.choice(candidates)
        return random.choice([v for v in choices if v != value])
    return random.choice(choices)


def mutate_replay(lines: list[str], idx: int) -> tuple[list[str], int]:
    current_frame = 0
    mutated = 0
    max_mutations = random.randint(MAX_TOTAL_MUTATIONS_MIN, MAX_TOTAL_MUTATIONS_MAX)
    out: list[str] = []

    for line in lines:
        fm = FRAME_RE.match(line)
        if fm:
            current_frame = int(fm.group(1))
            out.append(line)
            continue

        if FIGHTNAME_RE.match(line):
            out.append(f"FIGHTNAME 0; xioi_v{GENERATION}_{idx:02d}_early_focus")
            continue
        if AUTHOR_RE.match(line):
            out.append("AUTHOR 0; ToribashAI V30 early-focus evolution")
            continue

        jm = JOINT_RE.match(line)
        if jm and mutated < max_mutations:
            prefix, joint_s, value_s, suffix = jm.groups()
            joint = int(joint_s)
            value = int(value_s)
            rate = mutation_rate_for(current_frame, joint)

            if random.random() < rate:
                new_value = mutate_value(value)
                out.append(f"{prefix}{joint} {new_value}{suffix} # V30 mut {value}->{new_value}")
                mutated += 1
                continue

        out.append(line)

    # Si rien n'a muté, force 1 petite mutation early sur une ligne JOINT.
    if mutated == 0:
        out2 = []
        current_frame = 0
        candidates = []
        for n, line in enumerate(out):
            fm = FRAME_RE.match(line)
            if fm:
                current_frame = int(fm.group(1))
            jm = JOINT_RE.match(line)
            if jm and current_frame <= EARLY_END_FRAME and int(jm.group(2)) in EARLY_FOCUS_JOINTS:
                candidates.append(n)
        if candidates:
            n = random.choice(candidates)
            jm = JOINT_RE.match(out[n])
            assert jm
            prefix, joint_s, value_s, suffix = jm.groups()
            value = int(value_s)
            new_value = mutate_value(value)
            out[n] = f"{prefix}{joint_s} {new_value}{suffix} # V30 forced mut {value}->{new_value}"
            mutated = 1
        out2 = out
        out = out2

    return out, mutated


def clean_old_v30() -> None:
    for p in OUT_DIR.glob("xioi_v30_*.rpl"):
        p.unlink(missing_ok=True)
    for p in TORIBASH_REPLAY_DIR.glob("xioi_v30_*.rpl"):
        p.unlink(missing_ok=True)


def main() -> None:
    random.seed(RANDOM_SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TORIBASH_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    parent = find_parent()
    lines = read_lines(parent)

    clean_old_v30()

    parent_out = OUT_DIR / "xioi_v30_parent.rpl"
    write_lines(parent_out, lines)
    shutil.copy2(parent_out, TORIBASH_REPLAY_DIR / parent_out.name)

    print("Parent:", parent)
    print("Parent copied:", parent_out)

    made = []
    for i in range(1, NUM_MUTATIONS + 1):
        child_lines, nmut = mutate_replay(lines, i)
        out = OUT_DIR / f"xioi_v30_{i:02d}_mut.rpl"
        write_lines(out, child_lines)
        shutil.copy2(out, TORIBASH_REPLAY_DIR / out.name)
        made.append((out.name, nmut))

    print(f"\nGenerated {len(made)} V30 mutations and copied to Toribash replay dir:")
    print(TORIBASH_REPLAY_DIR)
    for name, nmut in made:
        print(f"  {name} mutations={nmut}")

    print("\nDans Toribash > Replays, teste xioi_v30_parent puis les xioi_v30_XX_mut.")
    print("Quand tu as une meilleure mutation :")
    print("  python3 scripts/promote_xioi_v30_best.py xioi_v30_XX_mut.rpl")


if __name__ == "__main__":
    main()
