#!/usr/bin/env python3
from pathlib import Path
import json
import random

PROJECT = Path.home() / "Documents" / "ToribashAI"

MOTIFS_PATH = PROJECT / "datasets" / "motifs" / "forward_motifs_v1.jsonl"
OUT_DIR = PROJECT / "models" / "walk_learning_v1"

NUM_CANDIDATES = 20
TOP_MOTIFS = 40
MOTIFS_PER_REPLAY = 2

MATCHFRAMES = 1200
TURNFRAMES = 20
MOD_NAME = "ToribashAI/toribashai_goal_flat_v1.tbm"

START_Z = 5.40
UKE_Y = 12.0


def load_motifs():
    motifs = []
    with MOTIFS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                motifs.append(json.loads(line))

    # On favorise les motifs qui avancent sans trop tomber
    motifs.sort(
        key=lambda m: (
            m.get("displacement_xy", 0)
            - abs(m.get("dz", 0)) * 0.8
            + m.get("speed_xy_per_frame", 0) * 50
        ),
        reverse=True,
    )
    return motifs[:TOP_MOTIFS]


def smooth_actions(actions):
    """
    Version plus prudente:
    - garde beaucoup de hold
    - évite trop de relax
    - réduit l'agressivité
    """
    out = []
    for v in actions:
        v = int(v)

        if v == 4:
            v = 3

        if v == 0:
            v = 3

        out.append(v)

    return out


def mutate_soft(actions, rng):
    out = list(actions)

    for i in range(len(out)):
        if rng.random() < 0.025:
            out[i] = rng.choice([1, 2, 3])

    return out


def actions_to_joint_line(player_id, actions):
    parts = []
    for jid, val in enumerate(actions):
        val = int(val)
        if val != 0:
            parts.append(str(jid))
            parts.append(str(val))

    return f"JOINT {player_id}; " + " ".join(parts)


def hold_all(player_id):
    return (
        f"JOINT {player_id}; "
        "0 3 1 3 2 3 3 3 4 3 5 3 6 3 7 3 8 3 9 3 "
        "10 3 11 3 12 3 13 3 14 3 15 3 16 3 17 3 18 3 19 3"
    )


def write_candidate(index, motifs):
    rng = random.Random(3000 + index)

    name = f"ToribashAI_walk_learning_{index:03d}"
    path = OUT_DIR / f"{name}.rpl"

    chosen = rng.choices(
        motifs,
        weights=[max(0.001, m.get("score", 0)) for m in motifs],
        k=MOTIFS_PER_REPLAY,
    )

    lines = []

    lines.append("#SCORE 0 0")
    lines.append("#WIN 2")
    lines.append("VERSION 12")
    lines.append(f"FIGHTNAME 0; {name}")
    lines.append("BOUT 0; ToribashAI")
    lines.append("BOUT 1; Target")
    lines.append("AUTHOR 0; ToribashAI")
    lines.append("AUTHOR 1; ToribashAI")

    lines.append(f"ENGAGE 0; 0.000000 0.000000 {START_Z:.6f} 0 0 0")
    lines.append(f"ENGAGE 1; 0.000000 {UKE_Y:.6f} {START_Z:.6f} 0 0 0")

    lines.append(
        f"NEWGAME 1;{MATCHFRAMES} {TURNFRAMES} 200000 0 0 8 200 0 1 "
        f"{MOD_NAME} 0 0 250 500 1000 0 1 0 2 0 0 0 0 0 0 "
        f"0.000000 0.000000 -30.000000 0 0 0 0 8"
    )

    frame = 0

    # Départ très stable
    lines.append(f"FRAME {frame}; 0 0 0 0")
    lines.append(hold_all(0))
    lines.append(hold_all(1))

    # 3 tours de stabilisation
    for _ in range(3):
        frame += TURNFRAMES
        lines.append(f"FRAME {frame}; 0 0 0 0")
        lines.append(hold_all(0))
        lines.append(hold_all(1))

    # Apprentissage marche: motifs humains ralentis et stabilisés
    for motif in chosen:
        for actions in motif["actions"]:
            frame += TURNFRAMES

            if frame >= MATCHFRAMES:
                break

            stable = smooth_actions(actions)
            stable = mutate_soft(stable, rng)

            lines.append(f"FRAME {frame}; 0 0 0 0")
            lines.append(actions_to_joint_line(0, stable))

            if frame % 100 == 0:
                lines.append(hold_all(1))

        # petite pause de stabilisation entre deux motifs
        for _ in range(2):
            frame += TURNFRAMES
            if frame >= MATCHFRAMES:
                break
            lines.append(f"FRAME {frame}; 0 0 0 0")
            lines.append(hold_all(0))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "path": str(path),
        "final_frame": frame,
        "motifs": [m["source_name"] for m in chosen],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in OUT_DIR.glob("*.rpl"):
        p.unlink()

    motifs = load_motifs()

    if not motifs:
        raise RuntimeError("Aucun motif trouvé. Lance d'abord extract_forward_motifs_v1.py")

    print(f"Motifs chargés: {len(motifs)}")
    print(f"Output: {OUT_DIR}")

    metadata = []

    for i in range(NUM_CANDIDATES):
        info = write_candidate(i, motifs)
        metadata.append(info)
        print(f"{Path(info['path']).name} | final_frame={info['final_frame']}")

    meta_path = OUT_DIR / "generation_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("Terminé.")
    print(f"Dossier: {OUT_DIR}")


if __name__ == "__main__":
    main()
