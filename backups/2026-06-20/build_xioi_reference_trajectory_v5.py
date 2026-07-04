#!/usr/bin/env python3
"""
build_xioi_reference_trajectory_v5.py

But:
    Créer une trajectoire de référence pour la branche walk_xioi_imitation.

Pourquoi:
    Jusqu'ici, l'évolution récompense surtout:
      - survivre
      - avancer
      - ne pas toucher le sol avec mains/tête/hanches/épaules

    Mais ça ne dit pas explicitement:
      "ressemble à la marche Xioi".

    Ce script transforme le seed Xioi en référence compacte:
      xioi_reference_trajectory_v1.json

Comment ça marche:
    Entrée:
      evolution/walk_xioi_imitation_seed_v1.json
      ou evolution/walk_xioi_seed_v1.json

    Le seed contient des commandes compressées:
      {"frame": 25, "pairs": [[14, 2], [15, 4], ...]}

    Le script reconstruit l'état des 20 joints pour chaque frame 0..427:
      frame 0 = état initial + commandes frame 0
      frame 1 = état précédent si aucune commande
      frame 25 = état précédent + modifications de frame 25

    Comme Python ne connaît pas la physique Toribash, cette V1 produit une référence
    "symbolique" à partir des joints:
      - joint_states[20]
      - active_count
      - pseudo bodies simplifiés

    Ensuite le Lua V3 peut charger ce fichier.
    Pour une V4 encore meilleure, on pourra créer une vraie référence physique en
    enregistrant les positions réelles depuis Toribash pendant une run Xioi propre.

Sortie:
    evolution/xioi_reference_trajectory_v1.json
"""

import json
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
EVOLUTION = ROOT / "evolution"

SOURCE_CANDIDATES = [
    EVOLUTION / "walk_xioi_imitation_seed_v1.json",
    EVOLUTION / "walk_xioi_seed_v1.json",
    EVOLUTION / "champion_xioi_mechanic_v7.json",
]

OUT = EVOLUTION / "xioi_reference_trajectory_v1.json"

LOOP_LENGTH = 428
JOINT_COUNT = 20
DEFAULT_STATE = 3

# This rough pose model is NOT real physics.
# It is only a bridge reference until we record true body positions from Toribash.
BODY_IDS = list(range(14))


def load_source():
    for path in SOURCE_CANDIDATES:
        if path.exists():
            print("Source:", path)
            return json.loads(path.read_text(encoding="utf-8")), path
    raise FileNotFoundError("Aucune source Xioi trouvée.")


def normalize_commands(data):
    commands = data.get("commands", [])
    clean = []

    for cmd in commands:
        if not isinstance(cmd, dict):
            continue

        frame = int(cmd.get("frame", 0))
        pairs = cmd.get("pairs", [])

        clean_pairs = []
        seen = set()

        for pair in pairs:
            if not isinstance(pair, list) or len(pair) < 2:
                continue
            joint = int(pair[0])
            state = int(pair[1])
            if 0 <= joint < JOINT_COUNT and 0 <= state <= 4 and joint not in seen:
                clean_pairs.append([joint, state])
                seen.add(joint)

        clean.append({"frame": frame, "pairs": clean_pairs})

    clean.sort(key=lambda c: c["frame"])
    return clean


def build_joint_frames(commands, loop_length):
    by_frame = {}
    for cmd in commands:
        by_frame.setdefault(int(cmd["frame"]), [])
        by_frame[int(cmd["frame"])].extend(cmd["pairs"])

    state = [DEFAULT_STATE] * JOINT_COUNT
    frames = []

    for frame in range(loop_length):
        for joint, value in by_frame.get(frame, []):
            state[joint] = int(value)

        frames.append(list(state))

    return frames


def pseudo_body_positions(joints, frame):
    """
    Very rough symbolic pose embedding.
    Body ids roughly:
      0 head, 1 chest, 2-5 hips/core, 6-9 shoulders/arms, 10-13 wrists/hands/glutes-ish.

    This gives Lua something posture-like to compare before we have true Toribash physics captures.
    """
    t = frame / LOOP_LENGTH

    # joint values centered around hold/relax-ish
    j = [(v - 3) for v in joints]

    # crude rhythmic baseline
    phase = (t * 2.0) - int(t * 2.0)

    bodies = {}

    center_x = frame * 0.015
    center_y = 0.0
    center_z = 6.4

    for body_id in BODY_IDS:
        x = center_x
        y = center_y
        z = center_z

        if body_id == 0:  # head
            z += 1.2 + 0.08 * j[0]
            y += 0.05 * j[1]
        elif body_id == 1:  # chest
            z += 0.55 + 0.08 * j[2]
            y += 0.05 * j[3]
        elif body_id in [2, 3, 4, 5]:  # core/hips
            z += 0.25 + 0.04 * (j[12] + j[13])
            y += 0.05 * (j[14] - j[15])
        elif body_id in [6, 7, 8, 9]:  # arms/shoulders
            side = -1 if body_id in [6, 8] else 1
            z += 0.45 + 0.06 * j[body_id % JOINT_COUNT]
            y += side * (0.55 + 0.04 * abs(j[body_id % JOINT_COUNT]))
            x += 0.10 * j[body_id % JOINT_COUNT]
        elif body_id in [10, 11]:  # hands/wrists
            side = -1 if body_id == 10 else 1
            z += 0.10 + 0.06 * j[body_id % JOINT_COUNT]
            y += side * 0.75
            x += 0.12 * j[body_id % JOINT_COUNT]
        else:  # lower/core placeholder
            side = -1 if body_id == 12 else 1
            z -= 0.35 + 0.08 * abs(j[body_id % JOINT_COUNT])
            y += side * 0.30
            x += 0.18 * j[body_id % JOINT_COUNT]

        bodies[str(body_id)] = {
            "x": round(x, 4),
            "y": round(y, 4),
            "z": round(z, 4),
        }

    # Store a synthetic center under -1; Lua can use it for relative normalization.
    bodies["-1"] = {
        "x": round(center_x, 4),
        "y": round(center_y, 4),
        "z": round(center_z, 4),
    }

    return bodies


def main():
    data, source_path = load_source()
    loop_length = int(data.get("loop_length", LOOP_LENGTH))

    commands = normalize_commands(data)
    joint_frames = build_joint_frames(commands, loop_length)

    reference_frames = []

    for frame, joints in enumerate(joint_frames):
        active_count = sum(1 for v in joints if v != 3)

        reference_frames.append(
            {
                "frame": frame,
                "joint_states": joints,
                "active_count": active_count,
                "bodies": pseudo_body_positions(joints, frame),
            }
        )

    out_obj = {
        "name": "xioi_reference_trajectory_v1",
        "branch": "walk_xioi_imitation",
        "source": str(source_path),
        "description": "Reference trajectory V1 reconstructed from Xioi command seed. Symbolic pseudo-body positions; later replace with true Toribash-captured trajectory.",
        "loop_length": loop_length,
        "joint_count": JOINT_COUNT,
        "body_ids": BODY_IDS,
        "frames": reference_frames,
    }

    OUT.write_text(json.dumps(out_obj, indent=2), encoding="utf-8")

    print("Référence écrite:", OUT)
    print("Frames:", len(reference_frames))
    print("Première frame:")
    print(json.dumps(reference_frames[0], indent=2)[:1200])


if __name__ == "__main__":
    main()
