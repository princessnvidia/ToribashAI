#!/usr/bin/env python3
"""
generate_xioi_template_mutations_v29.py

V29: mutations légères autour du template Xioi V28.
On garde tout le contexte du replay source (POS/QAT/ENGAGE/NEWGAME/etc.)
et on ne mute que quelques lignes JOINT, surtout au début.
"""
from __future__ import annotations

import random
import re
import shutil
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
GEN = ROOT / "generated_replays"
STEAM_REPLAY = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"

PARENT = GEN / "xioi_source_template_v28.rpl"
OUT_DIR = GEN / "xioi_v29_mutations"

POPULATION = 24
SEED = 29029

# Mutations très légères : on veut garder la beauté Xioi.
EARLY_FRAME_MAX = 140
MID_FRAME_MAX = 280
EARLY_MUTATION_RATE = 0.055
MID_MUTATION_RATE = 0.020
LATE_MUTATION_RATE = 0.006

# Probabilité de supprimer/ajouter très basse.
DROP_RATE = 0.008
ADD_RATE = 0.010

JOINT_RE = re.compile(r"^(JOINT\s+0;\s+)(\d+)\s+([1-4])\s*$")
FRAME_RE = re.compile(r"^FRAME\s+(\d+);?")

# Joints de marche importants : hanches/genoux/chevilles + bras simples.
WALK_JOINTS = [4,5,6,7,8,9,14,15,16,17,18,19]


def mutation_rate(frame: int) -> float:
    if frame <= EARLY_FRAME_MAX:
        return EARLY_MUTATION_RATE
    if frame <= MID_FRAME_MAX:
        return MID_MUTATION_RATE
    return LATE_MUTATION_RATE


def mutate_value(v: int) -> int:
    choices = [1, 2, 3, 4]
    # Mutation douce: préfère voisins, mais autorise parfois autre état.
    near = []
    if v > 1:
        near.append(v - 1)
    if v < 4:
        near.append(v + 1)
    if near and random.random() < 0.75:
        return random.choice(near)
    choices.remove(v)
    return random.choice(choices)


def mutate_lines(lines: list[str], idx: int) -> tuple[list[str], dict]:
    out = []
    current_frame = 0
    stats = {"changed": 0, "dropped": 0, "added": 0}

    for line in lines:
        mframe = FRAME_RE.match(line.strip())
        if mframe:
            current_frame = int(mframe.group(1))
            out.append(line)
            continue

        mj = JOINT_RE.match(line.strip())
        if mj:
            prefix, joint_s, val_s = mj.groups()
            joint = int(joint_s)
            val = int(val_s)
            rate = mutation_rate(current_frame)

            if random.random() < DROP_RATE and current_frame <= MID_FRAME_MAX:
                stats["dropped"] += 1
                continue

            if joint in WALK_JOINTS and random.random() < rate:
                new_val = mutate_value(val)
                out.append(f"{prefix}{joint} {new_val}\n")
                stats["changed"] += 1
            else:
                out.append(line)
            continue

        out.append(line)

        # Ajout rare juste après FRAME, pour tester micro corrections.
        if mframe and current_frame <= MID_FRAME_MAX and random.random() < ADD_RATE:
            j = random.choice(WALK_JOINTS)
            v = random.randint(1, 4)
            out.append(f"JOINT 0; {j} {v}\n")
            stats["added"] += 1

    # Renomme le fightname pour distinguer dans Toribash.
    renamed = []
    for line in out:
        if line.startswith("FIGHTNAME 0;"):
            renamed.append(f"FIGHTNAME 0; xioi_v29_mutation_{idx:02d}\n")
        elif line.startswith("AUTHOR 0;"):
            renamed.append("AUTHOR 0; ToribashAI V29\n")
        else:
            renamed.append(line)
    return renamed, stats


def main() -> None:
    random.seed(SEED)
    if not PARENT.exists():
        raise FileNotFoundError(f"Parent introuvable: {PARENT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STEAM_REPLAY.mkdir(parents=True, exist_ok=True)

    lines = PARENT.read_text(encoding="utf-8", errors="replace").splitlines(True)

    # Copie parent propre.
    parent_out = OUT_DIR / "xioi_v29_00_parent.rpl"
    parent_out.write_text("".join(lines), encoding="utf-8")
    shutil.copy2(parent_out, STEAM_REPLAY / parent_out.name)

    print("Parent:", parent_out)
    print("Mutations:")
    for i in range(1, POPULATION + 1):
        mutated, stats = mutate_lines(lines, i)
        path = OUT_DIR / f"xioi_v29_{i:02d}_mut.rpl"
        path.write_text("".join(mutated), encoding="utf-8")
        shutil.copy2(path, STEAM_REPLAY / path.name)
        print(f"  {path.name} changed={stats['changed']} dropped={stats['dropped']} added={stats['added']}")

    print("\nCopié dans Toribash replay:", STEAM_REPLAY)
    print("Teste les xioi_v29_* dans Setup > Replays, puis note les 3 meilleurs.")


if __name__ == "__main__":
    main()
