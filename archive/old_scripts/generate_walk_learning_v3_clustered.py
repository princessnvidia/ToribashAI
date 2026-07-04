#!/usr/bin/env python3
from pathlib import Path
import json
import random
import math
from collections import defaultdict

PROJECT = Path.home() / "Documents" / "ToribashAI"

MOTIFS_PATH = PROJECT / "datasets" / "motifs" / "forward_motifs_v1.jsonl"
OUT_DIR = PROJECT / "models" / "walk_learning_v3_clustered"

NUM_CANDIDATES = 60
TOP_MOTIFS = 600
MOTIFS_PER_REPLAY = 4

MATCHFRAMES = 1400
TURNFRAMES = 20
MOD_NAME = "ToribashAI/toribashai_goal_flat_v1.tbm"

START_Z = 5.40
UKE_Y = -12.0


def hold_all(player_id):
    return (
        f"JOINT {player_id}; "
        "0 3 1 3 2 3 3 3 4 3 5 3 6 3 7 3 8 3 9 3 "
        "10 3 11 3 12 3 13 3 14 3 15 3 16 3 17 3 18 3 19 3"
    )


def motif_signature(motif):
    actions = motif["actions"]
    active_counts = [sum(1 for v in a if int(v) != 0 and int(v) != 3) for a in actions]

    legs = []
    arms = []
    core = []

    for a in actions:
        legs.append(sum(1 for j in [14, 15, 16, 17, 18, 19] if int(a[j]) != 3))
        arms.append(sum(1 for j in [4, 5, 6, 7, 8, 9, 10, 11] if int(a[j]) != 3))
        core.append(sum(1 for j in [0, 1, 2, 3, 12, 13] if int(a[j]) != 3))

    return {
        "dy": motif.get("dy", 0.0),
        "dz": motif.get("dz", 0.0),
        "disp": motif.get("displacement_xy", 0.0),
        "avg_active": sum(active_counts) / max(1, len(active_counts)),
        "avg_legs": sum(legs) / max(1, len(legs)),
        "avg_arms": sum(arms) / max(1, len(arms)),
        "avg_core": sum(core) / max(1, len(core)),
    }


def cluster_key(sig):
    # Clustering simple: on regroupe les motifs qui ressemblent à de la locomotion
    # vers Y négatif, avec une activité jambes/bras comparable.
    dy_bucket = int(max(0, -sig["dy"]) // 3)
    leg_bucket = int(sig["avg_legs"] // 2)
    arm_bucket = int(sig["avg_arms"] // 2)
    active_bucket = int(sig["avg_active"] // 3)
    return (dy_bucket, leg_bucket, arm_bucket, active_bucket)


def load_clustered_motifs():
    motifs = []

    with MOTIFS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            m = json.loads(line)
            sig = motif_signature(m)

            # On garde uniquement des motifs plausibles de marche/course :
            # avance vers Y négatif, pas trop de chute, jambes actives.
            if sig["dy"] >= -1.0:
                continue
            if sig["dz"] < -10.0:
                continue
            if sig["avg_legs"] < 1.0:
                continue
            if sig["avg_active"] < 2.0:
                continue

            m["_sig"] = sig
            m["_cluster"] = cluster_key(sig)
            m["_walk_score"] = (
                max(0.0, -sig["dy"]) * 3.0
                + sig["disp"] * 0.8
                + sig["avg_legs"] * 1.5
                + sig["avg_arms"] * 0.5
                - abs(sig["dz"]) * 0.8
            )
            motifs.append(m)

    motifs.sort(key=lambda m: m["_walk_score"], reverse=True)
    motifs = motifs[:TOP_MOTIFS]

    clusters = defaultdict(list)
    for m in motifs:
        clusters[m["_cluster"]].append(m)

    # On garde les clusters avec plusieurs motifs similaires
    clusters = {k: v for k, v in clusters.items() if len(v) >= 3}

    if not clusters:
        raise RuntimeError("Aucun cluster de marche trouvé.")

    ranked_clusters = sorted(
        clusters.items(),
        key=lambda kv: sum(m["_walk_score"] for m in kv[1]) / len(kv[1]),
        reverse=True,
    )

    return ranked_clusters


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


def write_candidate(index, ranked_clusters):
    rng = random.Random(9000 + index)

    name = f"ToribashAI_walk_v3_clustered_{index:03d}"
    path = OUT_DIR / f"{name}.rpl"

    # Choisit UN cluster principal : les motifs d'un candidat se ressemblent.
    cluster_id, cluster = rng.choice(ranked_clusters[:12])

    chosen = rng.choices(
        cluster,
        weights=[max(0.001, m["_walk_score"]) for m in cluster],
        k=MOTIFS_PER_REPLAY,
    )

    mutation_rate = rng.uniform(0.005, 0.035)

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

    # Stabilisation courte
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

        if frame >= MATCHFRAMES:
            break

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "name": name,
        "path": str(path),
        "final_frame": frame,
        "cluster": str(cluster_id),
        "mutation_rate": mutation_rate,
        "motifs": [m["source_name"] for m in chosen],
        "dy": [m.get("dy", 0.0) for m in chosen],
        "walk_score": [m["_walk_score"] for m in chosen],
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for p in OUT_DIR.glob("*.rpl"):
        p.unlink()

    ranked_clusters = load_clustered_motifs()

    print(f"Clusters trouvés: {len(ranked_clusters)}")
    print(f"Output: {OUT_DIR}")

    metadata = []

    for i in range(NUM_CANDIDATES):
        info = write_candidate(i, ranked_clusters)
        metadata.append(info)
        print(
            f"{Path(info['path']).name} | "
            f"frame={info['final_frame']} | "
            f"mutation={info['mutation_rate']:.3f} | "
            f"cluster={info['cluster']}"
        )

    meta_path = OUT_DIR / "generation_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("Terminé.")
    print(f"Dossier: {OUT_DIR}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
