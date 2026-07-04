#!/usr/bin/env python3
from pathlib import Path
import json
import random

PROJECT = Path.home() / "Documents" / "ToribashAI"

DATA_PATH = PROJECT / "datasets" / "motion_patterns" / "forward_clean_v1.jsonl"
OUT_DIR = PROJECT / "models" / "walk_from_clean_forward_v1"

NUM_CANDIDATES = 80
PATTERNS_PER_REPLAY = 4

MATCHFRAMES = 1400
TURNFRAMES = 20
MOD_NAME = "ToribashAI/toribashai_goal_flat_v1.tbm"

START_Z = 5.40
UKE_Y = -12.0

MIN_Z = 4.5
MAX_Z = 30.0


def hold_all(player_id):
    return (
        f"JOINT {player_id}; "
        "0 3 1 3 2 3 3 3 4 3 5 3 6 3 7 3 8 3 9 3 "
        "10 3 11 3 12 3 13 3 14 3 15 3 16 3 17 3 18 3 19 3"
    )


def load_patterns():
    rows = []

    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            feat = row["features"]
            conf = row["classifier"]["target_confidence"]

            z_min = float(feat["z_min"])
            dy = float(feat["delta_y"])

            if z_min < MIN_Z or z_min > MAX_Z:
                continue

            if dy > -2.0:
                continue

            row["_gen_score"] = (
                abs(dy) * 3.0
                + float(conf) * 10.0
                + float(feat["leg_activity"]) * 3.0
                + float(feat["arm_activity"]) * 1.0
                - abs(float(feat["delta_z"])) * 0.8
            )

            rows.append(row)

    rows.sort(key=lambda r: r["_gen_score"], reverse=True)
    return rows


def normalize_action_value(v):
    v = int(v)
    if v == 0:
        return 3
    if v == 4:
        return 3
    return max(1, min(3, v))


def mutate_actions(actions, rng, rate):
    out = [normalize_action_value(v) for v in actions]

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


def write_candidate(index, patterns):
    rng = random.Random(12000 + index)

    name = f"ToribashAI_clean_walk_{index:03d}"
    path = OUT_DIR / f"{name}.rpl"

    mutation_rate = rng.uniform(0.002, 0.025)

    chosen = rng.choices(
        patterns[:600],
        weights=[max(0.001, p["_gen_score"]) for p in patterns[:600]],
        k=PATTERNS_PER_REPLAY,
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

    # Stabilisation
    for _ in range(2):
        frame += TURNFRAMES
        lines.append(f"FRAME {frame}; 0 0 0 0")
        lines.append(hold_all(0))
        lines.append(hold_all(1))

    for pattern in chosen:
        for actions in pattern["actions"]:
            frame += TURNFRAMES
            if frame >= MATCHFRAMES:
                break

            a = mutate_actions(actions, rng, mutation_rate)
            lines.append(f"FRAME {frame}; 0 0 0 0")
            lines.append(actions_to_joint_line(0, a))

        if frame >= MATCHFRAMES:
            break

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "name": name,
        "path": str(path),
        "final_frame": frame,
        "mutation_rate": mutation_rate,
        "patterns": [
            {
                "source_name": p["source_name"],
                "start_frame": p["start_frame"],
                "end_frame": p["end_frame"],
                "delta_y": p["features"]["delta_y"],
                "z_min": p["features"]["z_min"],
                "confidence": p["classifier"]["target_confidence"],
                "gen_score": p["_gen_score"],
            }
            for p in chosen
        ],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in OUT_DIR.glob("*.rpl"):
        p.unlink()

    patterns = load_patterns()

    if not patterns:
        raise RuntimeError("Aucun pattern propre trouvé.")

    print(f"Patterns propres chargés: {len(patterns)}")
    print(f"Output: {OUT_DIR}")

    metadata = []

    for i in range(NUM_CANDIDATES):
        info = write_candidate(i, patterns)
        metadata.append(info)
        print(
            f"{Path(info['path']).name} | "
            f"frame={info['final_frame']} | "
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
