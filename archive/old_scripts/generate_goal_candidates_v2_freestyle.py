#!/usr/bin/env python3
from pathlib import Path
import random

PROJECT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = PROJECT / "models" / "goal_candidates_v2_freestyle"

NUM_CANDIDATES = 40
MATCHFRAMES = 2000
TURNFRAMES = 10
MOD_NAME = "ToribashAI/toribashai_goal_flat_v1.tbm"

# Joints Toribash classiques approximatifs
JOINTS = list(range(20))

# Valeurs Toribash:
# 1 = extend
# 2 = contract
# 3 = hold
# 4 = relax

BASE_FRAMES = list(range(0, 900, TURNFRAMES))


def clamp_action(v):
    return max(1, min(4, int(v)))


def make_joint_line(player, actions):
    parts = []
    for joint_id, value in sorted(actions.items()):
        parts.append(str(joint_id))
        parts.append(str(clamp_action(value)))

    if not parts:
        return None

    return f"JOINT {player}; " + " ".join(parts)


def gait_pattern(frame, rng, aggression=0.5, lean=0.5, asymmetry=0.0):
    """
    Générateur freestyle orienté course vers la cible.
    Il ne copie pas un replay: il fabrique des cycles de marche/course.
    """

    t = frame // TURNFRAMES
    phase = t % 8

    a = {}

    # Base: garder un peu le corps contrôlé
    for j in [0, 1, 2, 3]:
        if rng.random() < 0.45:
            a[j] = 3

    # Cycle jambes alternées
    left_phase = phase in [0, 1, 2, 3]
    right_phase = not left_phase

    # Hanches / genoux / chevilles approximatifs
    # On crée un mouvement alterné plutôt qu'une copie.
    if left_phase:
        a[15] = 2 if rng.random() < 0.75 else 3
        a[17] = 1 if rng.random() < 0.70 else 3
        a[19] = 2 if rng.random() < 0.55 else 3

        a[14] = 1 if rng.random() < 0.65 else 3
        a[16] = 2 if rng.random() < 0.65 else 3
        a[18] = 1 if rng.random() < 0.50 else 3
    else:
        a[14] = 2 if rng.random() < 0.75 else 3
        a[16] = 1 if rng.random() < 0.70 else 3
        a[18] = 2 if rng.random() < 0.55 else 3

        a[15] = 1 if rng.random() < 0.65 else 3
        a[17] = 2 if rng.random() < 0.65 else 3
        a[19] = 1 if rng.random() < 0.50 else 3

    # Bras opposés aux jambes pour stabiliser
    if left_phase:
        a[4] = 1
        a[5] = 2
        a[6] = 2
        a[7] = 1
    else:
        a[4] = 2
        a[5] = 1
        a[6] = 1
        a[7] = 2

    # Lean / impulsion vers l'avant
    if rng.random() < lean:
        a[2] = 2
        a[3] = 2

    # Plus d'agression = plus de relax/impulsion
    if rng.random() < aggression:
        for j in rng.sample(JOINTS, rng.randint(1, 4)):
            a[j] = rng.choice([1, 2, 4])

    # Asymétrie légère pour explorer
    if rng.random() < abs(asymmetry):
        side_joints = [4, 6, 8, 10, 14, 16, 18] if asymmetry > 0 else [5, 7, 9, 11, 15, 17, 19]
        for j in rng.sample(side_joints, rng.randint(1, 3)):
            a[j] = rng.choice([1, 2, 3, 4])

    # Mutation ponctuelle
    if rng.random() < 0.18:
        for _ in range(rng.randint(1, 3)):
            a[rng.choice(JOINTS)] = rng.choice([1, 2, 3, 4])

    return a


def write_candidate(index):
    rng = random.Random(1000 + index)

    aggression = rng.uniform(0.15, 0.85)
    lean = rng.uniform(0.35, 0.9)
    asymmetry = rng.uniform(-0.35, 0.35)

    name = f"ToribashAI_goal_freestyle_{index:03d}"
    path = OUT_DIR / f"{name}.rpl"

    lines = []

    lines.append("#SCORE 0 0")
    lines.append("#WIN 2")
    lines.append("VERSION 12")
    lines.append(f"FIGHTNAME 0; {name}")
    lines.append("BOUT 0; ToribashAI")
    lines.append("BOUT 1; Target")
    lines.append("AUTHOR 0; ToribashAI")
    lines.append("AUTHOR 1; ToribashAI")
    lines.append("ENGAGE 0; 0.000000 0.000000 12.600000 0 0 0")
    lines.append("ENGAGE 1; 0.000000 24.000000 12.600000 0 0 0")
    lines.append(
        f"NEWGAME 1;{MATCHFRAMES} {TURNFRAMES} 200000 0 0 8 200 0 1 "
        f"{MOD_NAME} 0 0 250 500 1000 0 1 0 2 0 0 0 0 0 0 "
        f"0.000000 0.000000 -30.000000 0 0 0 0 8"
    )

    # Départ: posture plutôt stable
    lines.append("FRAME 0; 0 0 0 0")
    lines.append("JOINT 0; 0 3 1 3 2 3 3 3 4 3 5 3 6 3 7 3 8 3 9 3 10 3 11 3 12 3 13 3 14 3 15 3 16 3 17 3 18 3 19 3")
    lines.append("JOINT 1; 0 3 1 3 2 3 3 3 4 3 5 3 6 3 7 3 8 3 9 3 10 3 11 3 12 3 13 3 14 3 15 3 16 3 17 3 18 3 19 3")

    for frame in BASE_FRAMES[1:]:
        actions = gait_pattern(frame, rng, aggression=aggression, lean=lean, asymmetry=asymmetry)

        lines.append(f"FRAME {frame}; 0 0 0 0")

        line0 = make_joint_line(0, actions)
        if line0:
            lines.append(line0)

        # Uke reste immobile
        if frame % 80 == 0:
            lines.append("JOINT 1; 0 3 1 3 2 3 3 3 4 3 5 3 6 3 7 3 8 3 9 3 10 3 11 3 12 3 13 3 14 3 15 3 16 3 17 3 18 3 19 3")

        # Grip parfois
        if rng.random() < 0.04:
            lines.append(f"GRIP 0; {rng.choice([0, 1, 2])} {rng.choice([0, 1, 2])}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "path": path,
        "aggression": aggression,
        "lean": lean,
        "asymmetry": asymmetry,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    old = list(OUT_DIR.glob("*.rpl"))
    for p in old:
        p.unlink()

    print(f"Output: {OUT_DIR}")
    print(f"Generating {NUM_CANDIDATES} freestyle candidates...")

    infos = []
    for i in range(NUM_CANDIDATES):
        info = write_candidate(i)
        infos.append(info)
        print(
            f"{info['path'].name} | "
            f"aggression={info['aggression']:.2f} "
            f"lean={info['lean']:.2f} "
            f"asymmetry={info['asymmetry']:.2f}"
        )

    print()
    print("Terminé.")
    print("Ces replays sont actions-only: Toribash doit recalculer la physique en les lançant.")
    print(f"Dossier: {OUT_DIR}")


if __name__ == "__main__":
    main()
