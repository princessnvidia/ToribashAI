#!/usr/bin/env python3
from pathlib import Path
import json
import random

PROJECT = Path.home() / "Documents" / "ToribashAI"

MOTIFS_PATH = PROJECT / "datasets" / "motifs" / "forward_motifs_v1.jsonl"
OUT_DIR = PROJECT / "models" / "walk_learning_v2"

NUM_CANDIDATES = 100
TOP_MOTIFS = 120
MOTIFS_PER_REPLAY = 3

MATCHFRAMES = 1400
TURNFRAMES = 20
MOD_NAME = "ToribashAI/toribashai_goal_flat_v1.tbm"

START_Z = 5.40
UKE_Y = -12.0  # on suit la direction naturelle trouvée: Y négatif


def load_motifs():
    motifs = []
    with MOTIFS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                motifs.append(json.loads(line))

    # Favorise les motifs qui avancent vers Y négatif sans trop tomber
    motifs.sort(
        key=lambda m: (
            max(0.0, -m.get("dy", 0.0)) * 2.0
            + m.get("displacement_xy", 0.0)
            - abs(m.get("dz", 0.0)) * 0.7
            + m.get("speed_xy_per_frame", 0.0) * 60.0
        ),
        reverse=True,
    )
    return motifs[:TOP_MOTIFS]


def hold_all(player_id):
    return (
        f"JOINT {player_id}; "
        "0 3 1 3 2 3 3 3 4 3 5 3 6 3 7 3 8 3 9 3 "
        "10 3 11 3 12 3 13 3 14 3 15 3 16 3 17 3 18 3 19 3"
    )


def smooth_actions(actions):
    out = []
    for v in actions:
        v = int(v)
        if v == 0:
            v = 3
        if v == 4:
            v = 3
        out.append(v)
    return out


def mutate_actions(actions, rng, rate):
    out = list(actions)
    for i in range(len(out)):
        if rng.random() < rate:
            out[i] = rng.choice([1, 2, 3])
    return out


def actions_to_joint_line(player_id, actions):
    parts = []
    for jid, val in enumerate(actions):
        parts.append(str(jid))
        parts.append(str(int(val)))
    return f"JOINT {player_id}; " + " ".join(parts)


def write_candidate(index, motifs):
    rng = random.Random(5000 + index)

    name = f"ToribashAI_walk_v2_{index:03d}"
    path = OUT_DIR / f"{name}.rpl"

    mutation_rate = rng.uniform(0.015, 0.08)

    chosen = rng.choices(
        motifs,
        weights=[max(0.001, max(0.0, -m.get("dy", 0.0)) + m.get("score", 0.0) * 0.1) for m in motifs],
        k=MOTIFS_PER_REPLAY,
    )

    lines = [
        "#SCORE 0 0",
        "#WIN 2",
        "VERSION 12",
        f"FIGHTNAME 0; {name}",
        "BOUT 0; ToribashAI",
        "BOUT 1; Target",
        "AUTHOR 0; ToribashAI",
        "AUTHOR 1; ToribashAI",
        f"ENGAGE 0; 0.000000 0.000000 {START_Z:.6f} 0 0 0",
        f"ENGAGE 1; 0.000000 {UKE_Y:.6f} {START_Z:.6f} 0 0 0",
        (
            f"NEWGAME 1;{MATCHFRAMES} {TURNFRAMES} 200000 0 0 8 200 0 1 "
            f"{MOD_NAME} 0 0 250 500 1000 0 1 0 2 0 0 0 0 0 0 "
            f"0.000000 0.000000 -30.000000 0 0 0 0 8"
        ),
    ]

    frame = 0
    lines.append(f"FRAME {frame}; 0 0 0 0")
    lines.append(hold_all(0))
    lines.append(hold_all(1))

    # stabilisation initiale
    for _ in range(2):
        frame += TURNFRAMES
        lines.append(f"FRAME {frame}; 0 0 0 0")
        lines.append(hold_all(0))
        lines.append(hold_all(1))

    for motif in chosen:
        for actions in motif["actions"]:
            frame += TURNFRAMES
            if frame >= MATCHFRAMES:
                break

            a = smooth_actions(actions)
            a = mutate_actions(a, rng, mutation_rate)

            lines.append(f"FRAME {frame}; 0 0 0 0")
            lines.append(actions_to_joint_line(0, a))

        # micro pause pour éviter les chutes trop violentes
        if rng.random() < 0.45:
            frame += TURNFRAMES
            if frame < MATCHFRAMES:
                lines.append(f"FRAME {frame}; 0 0 0 0")
                lines.append(hold_all(0))

        if frame >= MATCHFRAMES:
            break

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "name": name,
        "path": str(path),
        "final_frame": frame,
        "mutation_rate": mutation_rate,
        "motifs": [m["source_name"] for m in chosen],
        "motif_dy": [m.get("dy", 0.0) for m in chosen],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in OUT_DIR.glob("*.rpl"):
        p.unlink()

    motifs = load_motifs()
    if not motifs:
        raise RuntimeError("Aucun motif trouvé. Lance extract_forward_motifs_v1.py")

    print(f"Motifs chargés: {len(motifs)}")
    print(f"Output: {OUT_DIR}")

    metadata = []

    for i in range(NUM_CANDIDATES):
        info = write_candidate(i, motifs)
        metadata.append(info)
        print(
            f"{Path(info['path']).name} | "
            f"final_frame={info['final_frame']} | "
            f"mutation={info['mutation_rate']:.3f}"
        )

    meta_path = OUT_DIR / "generation_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("Terminé.")
    print(f"Dossier: {OUT_DIR}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
