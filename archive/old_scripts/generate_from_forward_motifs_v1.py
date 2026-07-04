#!/usr/bin/env python3
from pathlib import Path
import json
import random

PROJECT = Path.home() / "Documents" / "ToribashAI"

MOTIFS_PATH = PROJECT / "datasets" / "motifs" / "forward_motifs_v1.jsonl"
OUT_DIR = PROJECT / "models" / "goal_candidates_from_motifs_v1"

NUM_CANDIDATES = 30
TOP_MOTIFS = 80
MOTIFS_PER_REPLAY = 6

MATCHFRAMES = 2000
TURNFRAMES = 10
MOD_NAME = "ToribashAI/toribashai_goal_flat_v1.tbm"


def load_motifs():
    motifs = []
    with MOTIFS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                motifs.append(json.loads(line))

    motifs.sort(key=lambda m: m.get("score", 0), reverse=True)
    return motifs[:TOP_MOTIFS]


def actions_to_joint_line(player_id, actions):
    parts = []
    for jid, val in enumerate(actions):
        val = int(val)
        if val != 0:
            parts.append(str(jid))
            parts.append(str(val))

    if not parts:
        return None

    return f"JOINT {player_id}; " + " ".join(parts)


def mutate_actions(actions, rng, mutation_rate=0.06):
    out = list(actions)

    for i in range(len(out)):
        if rng.random() < mutation_rate:
            out[i] = rng.choice([1, 2, 3, 4])

    return out


def write_candidate(index, motifs):
    rng = random.Random(2000 + index)

    name = f"ToribashAI_motif_goal_{index:03d}"
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
    lines.append("ENGAGE 0; 0.000000 0.000000 5.400000 0 0 0")
    lines.append("ENGAGE 1; 0.000000 12.000000 5.400000 0 0 0")

    lines.append(
        f"NEWGAME 1;{MATCHFRAMES} {TURNFRAMES} 200000 0 0 8 200 0 1 "
        f"{MOD_NAME} 0 0 250 500 1000 0 1 0 2 0 0 0 0 0 0 "
        f"0.000000 0.000000 -30.000000 0 0 0 0 8"
    )

    frame = 0

    lines.append(f"FRAME {frame}; 0 0 0 0")
    lines.append(
        "JOINT 0; "
        "0 3 1 3 2 3 3 3 4 3 5 3 6 3 7 3 8 3 9 3 "
        "10 3 11 3 12 3 13 3 14 3 15 3 16 3 17 3 18 3 19 3"
    )
    lines.append(
        "JOINT 1; "
        "0 3 1 3 2 3 3 3 4 3 5 3 6 3 7 3 8 3 9 3 "
        "10 3 11 3 12 3 13 3 14 3 15 3 16 3 17 3 18 3 19 3"
    )

    for motif in chosen:
        for actions in motif["actions"]:
            frame += TURNFRAMES

            if frame >= MATCHFRAMES:
                break

            mutated = mutate_actions(actions, rng)

            lines.append(f"FRAME {frame}; 0 0 0 0")

            joint_line = actions_to_joint_line(0, mutated)
            if joint_line:
                lines.append(joint_line)

            if frame % 100 == 0:
                lines.append(
                    "JOINT 1; "
                    "0 3 1 3 2 3 3 3 4 3 5 3 6 3 7 3 8 3 9 3 "
                    "10 3 11 3 12 3 13 3 14 3 15 3 16 3 17 3 18 3 19 3"
                )

        if frame >= MATCHFRAMES:
            break

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "path": str(path),
        "motifs": [m["source_name"] for m in chosen],
        "final_frame": frame,
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
    print(f"Metadata: {meta_path}")
    print()
    print("Vérifie qu'il n'y a pas de physique copiée avec:")
    print(f"grep \"^POS\\|^QAT\\|^LINVEL\\|^ANGVEL\" {OUT_DIR}/*.rpl | head")


if __name__ == "__main__":
    main()
