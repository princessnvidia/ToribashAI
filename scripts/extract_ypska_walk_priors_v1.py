#!/usr/bin/env python3
"""
extract_ypska_walk_priors_v1.py

Transforme le JSON YpSkA / loop 195 frames en priors utilisables par root_walk_v1.

But:
- garder le launch 0->126 comme démonstration forte
- garder 127->195 comme teacher de transition/marche
- créer des templates gauche/droite par miroir, pour éviter d'apprendre un seul côté
- produire evolution/root_walk_priors_v1.json

Usage:
  cd ~/Documents/ToribashAI
  python3 scripts/extract_ypska_walk_priors_v1.py \
    --input evolution/ypska_launch_0_195_commands.json \
    --output evolution/root_walk_priors_v1.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

JOINT_NAMES = {
    0: "neck",
    1: "chest",
    2: "lumbar",
    3: "abs",
    4: "right_pec",
    5: "right_shoulder",
    6: "right_elbow",
    7: "left_pec",
    8: "left_shoulder",
    9: "left_elbow",
    10: "right_wrist",
    11: "left_wrist",
    12: "right_glute",
    13: "left_glute",
    14: "right_hip",
    15: "left_hip",
    16: "right_knee",
    17: "left_knee",
    18: "right_ankle",
    19: "left_ankle",
}

MIRROR_JOINT = {
    4: 7, 7: 4,
    5: 8, 8: 5,
    6: 9, 9: 6,
    10: 11, 11: 10,
    12: 13, 13: 12,
    14: 15, 15: 14,
    16: 17, 17: 16,
    18: 19, 19: 18,
}

CONTROL_JOINTS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 14, 15, 16, 17, 18, 19]


def load_commands(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        raw = data.get("commands") or data.get("frames") or data.get("actions")
    elif isinstance(data, list):
        raw = data
    else:
        raw = None

    if not isinstance(raw, list):
        raise SystemExit(f"Impossible de trouver commands/frames/actions dans {path}")

    commands: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        frame = item.get("frame", item.get("t", item.get("index")))
        pairs = item.get("pairs", item.get("joints", item.get("command", item.get("actions"))))
        if frame is None or pairs is None:
            continue

        normalized_pairs: list[list[int]] = []
        if isinstance(pairs, dict):
            iterable = pairs.items()
        else:
            iterable = pairs

        for pair in iterable:
            if isinstance(pair, (list, tuple)) and len(pair) >= 2:
                k, v = pair[0], pair[1]
            else:
                continue
            try:
                j = int(k)
                val = int(v)
            except Exception:
                continue
            normalized_pairs.append([j, val])

        commands.append({"frame": int(frame), "pairs": normalized_pairs})

    commands.sort(key=lambda x: x["frame"])
    return commands


def slice_commands(commands: list[dict[str, Any]], start: int, end: int) -> list[dict[str, Any]]:
    out = []
    for c in commands:
        if start <= c["frame"] <= end:
            out.append({"frame": c["frame"], "pairs": [p[:] for p in c["pairs"]]})
    return out


def mirror_commands(commands: list[dict[str, Any]], base_start: int | None = None) -> list[dict[str, Any]]:
    """
    Miroir gauche/droite des joints. On garde les frames identiques par défaut.
    """
    out = []
    for c in commands:
        pairs = []
        for j, v in c["pairs"]:
            pairs.append([MIRROR_JOINT.get(j, j), v])
        out.append({"frame": c["frame"] if base_start is None else base_start + (c["frame"] - commands[0]["frame"]), "pairs": pairs})
    return out


def active_stats(commands: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    counts = Counter()
    values = defaultdict(Counter)
    for c in commands:
        if not (start <= c["frame"] <= end):
            continue
        for j, v in c["pairs"]:
            counts[j] += 1
            values[j][v] += 1

    return {
        "frame_range": [start, end],
        "joint_counts": {
            str(j): {
                "name": JOINT_NAMES.get(j, f"joint_{j}"),
                "count": counts[j],
                "values": {str(k): v for k, v in sorted(values[j].items())},
            }
            for j in sorted(counts)
        },
    }


def compress_to_last_state(commands: list[dict[str, Any]], start: int, end: int) -> dict[str, int]:
    """
    Résume une phase par dernier état connu par joint.
    Ce n'est pas un replay: c'est un état de skill de départ.
    """
    state: dict[int, int] = {}
    for c in commands:
        if start <= c["frame"] <= end:
            for j, v in c["pairs"]:
                if j in CONTROL_JOINTS:
                    state[j] = v
    return {str(k): v for k, v in sorted(state.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="evolution/ypska_launch_0_195_commands.json")
    ap.add_argument("--output", default="evolution/root_walk_priors_v1.json")
    ap.add_argument("--launch-end", type=int, default=126)
    ap.add_argument("--transition-end", type=int, default=170)
    ap.add_argument("--loop-end", type=int, default=195)
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    commands = load_commands(in_path)

    launch = slice_commands(commands, 0, args.launch_end)
    transition = slice_commands(commands, args.launch_end + 1, args.transition_end)
    walk_seed = slice_commands(commands, args.transition_end + 1, args.loop_end)

    # Templates simples: on part de la fin de transition/loop, puis on miroir.
    # L'évolution affinera ensuite.
    step_left_state = compress_to_last_state(commands, 127, args.transition_end)
    step_right_state = {str(MIRROR_JOINT.get(int(j), int(j))): v for j, v in step_left_state.items()}
    step_right_state = {str(k): step_right_state[str(k)] for k in sorted(map(int, step_right_state.keys()))}

    launch_stats = active_stats(commands, 0, args.launch_end)
    transition_stats = active_stats(commands, args.launch_end + 1, args.transition_end)
    walk_seed_stats = active_stats(commands, args.transition_end + 1, args.loop_end)

    priors = {
        "name": "root_walk_priors_v1",
        "source": str(in_path),
        "schema": 1,
        "principle": "YpSkA/loop sert de professeur: launch hard-freeze, transition soft teacher, marche par skills dynamiques.",
        "ranges": {
            "launch": [0, args.launch_end],
            "transition": [args.launch_end + 1, args.transition_end],
            "walk_seed": [args.transition_end + 1, args.loop_end],
        },
        "joint_names": {str(k): v for k, v in JOINT_NAMES.items()},
        "mirror_joint": {str(k): v for k, v in MIRROR_JOINT.items()},
        "launch": {
            "freeze": "hard",
            "commands": launch,
            "stats": launch_stats,
        },
        "transition_teacher": {
            "freeze": "soft",
            "commands": transition,
            "mirrored_commands": mirror_commands(transition),
            "stats": transition_stats,
        },
        "walk_teacher": {
            "freeze": "teacher_only",
            "commands": walk_seed,
            "mirrored_commands": mirror_commands(walk_seed),
            "stats": walk_seed_stats,
        },
        "skill_priors": {
            "step_left": step_left_state,
            "step_right": step_right_state,
            "balance": {
                "1": 3, "2": 3, "3": 3,
                "4": 3, "5": 3, "6": 2,
                "7": 3, "8": 3, "9": 2,
                "12": 3, "13": 3,
                "14": 3, "15": 3, "16": 3, "17": 3, "18": 3, "19": 3,
            },
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(priors, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {out_path}")
    print("Launch commands:", len(launch))
    print("Transition commands:", len(transition))
    print("Walk seed commands:", len(walk_seed))
    print("\nTop launch joints:")
    counts = []
    for j, info in launch_stats["joint_counts"].items():
        counts.append((info["count"], int(j), info["name"], info["values"]))
    for count, j, name, values in sorted(counts, reverse=True)[:12]:
        print(f"  {j:2d} {name:16s} count={count:3d} values={values}")


if __name__ == "__main__":
    main()
