#!/usr/bin/env python3
"""
make_root_walk_seed_v1.py

Crée root_walk_seed_v1.json et root_walk_champion_v1.json depuis root_walk_priors_v1.json.

La V1 utilise:
- launch YpSkA hard-freeze jusqu'à 126
- transition/loop comme professeur, pas comme replay permanent
- skills step_left/step_right initialisés depuis la démo + miroir
- bras complets: pecs, épaules, coudes
- score hybride imitation -> physique

Usage:
  cd ~/Documents/ToribashAI
  python3 scripts/make_root_walk_seed_v1.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--priors", default="evolution/root_walk_priors_v1.json")
    ap.add_argument("--seed", default="evolution/root_walk_seed_v1.json")
    ap.add_argument("--champion", default="evolution/root_walk_champion_v1.json")
    ap.add_argument("--state", default="evolution/root_walk_curriculum_state.json")
    args = ap.parse_args()

    priors_path = Path(args.priors)
    if not priors_path.exists():
        raise SystemExit(f"Missing {priors_path}. Lance d'abord extract_ypska_walk_priors_v1.py")

    priors = load_json(priors_path)
    skill_priors = priors.get("skill_priors", {})

    def skill(name: str, fallback: dict[str, int]) -> dict[str, int]:
        s = skill_priors.get(name)
        if isinstance(s, dict) and s:
            out = {str(k): int(v) for k, v in s.items()}
            for k, v in fallback.items():
                out.setdefault(str(k), int(v))
            return dict(sorted(out.items(), key=lambda kv: int(kv[0])))
        return {str(k): int(v) for k, v in fallback.items()}

    # États de base. Les valeurs exactes seront mutées.
    balance = skill("balance", {
        "1": 3, "2": 3, "3": 3,
        "4": 3, "5": 3, "6": 2,
        "7": 3, "8": 3, "9": 2,
        "12": 3, "13": 3,
        "14": 3, "15": 3, "16": 3, "17": 3, "18": 3, "19": 3,
    })

    step_left = skill("step_left", {
        # bras droit en contrepoids / bras gauche compact, jambes inversées par rapport à step_right
        "1": 3, "2": 3, "3": 3,
        "4": 2, "5": 2, "6": 2,
        "7": 4, "8": 4, "9": 2,
        "12": 3, "13": 3,
        "14": 4, "15": 2, "16": 4, "17": 2, "18": 3, "19": 3,
    })

    step_right = skill("step_right", {
        "1": 3, "2": 3, "3": 3,
        "4": 4, "5": 4, "6": 2,
        "7": 2, "8": 2, "9": 2,
        "12": 3, "13": 3,
        "14": 2, "15": 4, "16": 2, "17": 4, "18": 3, "19": 3,
    })

    agent = {
        "name": "root_walk_seed_v1",
        "branch": "root_walk_v1",
        "schema": 1,
        "description": (
            "Root walk v1: launch/loop YpSkA comme teacher, puis marche par perception, "
            "appuis, bras contralatéraux et skills mutables."
        ),

        "source_priors": str(priors_path),
        "schedule": [
            {"skill": "launch_ypska", "start_frame": 0, "end_frame": 126, "freeze": "hard"},
            {"skill": "settle_after_launch", "start_frame": 127, "end_frame": 170, "freeze": "soft_teacher"},
            {"skill": "root_gait_controller", "start_frame": 171, "end_frame": 1000, "freeze": "none"},
        ],

        "launch": {
            "hard_freeze_until": 126,
            "soft_teacher_until": 170,
            "commands": priors["launch"]["commands"],
            "transition_teacher": priors.get("transition_teacher", {}).get("commands", []),
            "target_min_velocity_y": 0.06,
            "target_min_hip_z": 2.3,
            "target_min_chest_z": 2.8,
        },

        "controller": {
            "max_frames": 320,
            "debug": True,
            "teacher_imitation_weight": 0.65,
            "physics_weight": 0.35,
            "teacher_decay_after_generations": 12,

            # V1.1: garde la loop/transition comme échafaudage moteur pendant toute
            # l'évaluation courte. Sans ça, le launch se termine puis les skills
            # statiques peuvent laisser Tori s'éteindre.
            "teacher_walk_overlay": True,
            "teacher_overlay_until": 420,
            "force_teacher_motor_until": 420,
            "teacher_loop_start": 127,
            "teacher_loop_end": 194,

            "foot_contact_z": 0.70,
            "hand_contact_z": 0.85,
            "knee_contact_z": 0.75,
            "shoulder_contact_z": 1.20,

            "min_head_z": 2.8,
            "min_chest_z": 2.35,
            "min_hip_z": 1.85,
            "stable_hip_z": 2.55,

            "min_step_forward": 0.12,
            "min_phase_frames": 8,
            "max_phase_frames": 34,
            "push_frames": 8,
            "speed_bonus_after_alternating_steps": 2,

            "max_side_tilt": 0.80,
            "max_forward_drop": 1.15,
        },

        "body_ids": {
            "note": "À ajuster si les métriques Z/Y semblent absurdes dans le result JSON.",
            "head": 0,
            "chest": 1,
            "hip": 4,
            "left_foot": 19,
            "right_foot": 18,
            "left_hand": 11,
            "right_hand": 10,
            "left_knee": 17,
            "right_knee": 16,
            "left_shoulder": 8,
            "right_shoulder": 5,
        },

        "arms": {
            "enabled": True,
            "used_for_launch": True,
            "used_for_counterbalance": True,
            "used_for_recovery": True,
            "elbows_closed_during_walk": True,
            "right_elbow_closed": 2,
            "left_elbow_closed": 2,
            "hand_min_z": 1.05,
            "max_pec_diff": 0.50,
            "max_shoulder_drop": 0.55,
            "arm_drive_strength": 1.0,
            "elbow_compact_weight": 0.7,
            "hand_ground_penalty": 120,
        },

        "skills": {
            "balance": balance,

            "step_left": step_left,
            "step_right": step_right,

            "push_left": {
                "1": 3, "2": 3, "3": 3,
                "4": 2, "5": 3, "6": 2,
                "7": 4, "8": 3, "9": 2,
                "12": 3, "13": 4,
                "14": 3, "15": 4, "16": 3, "17": 4, "18": 3, "19": 4,
            },

            "push_right": {
                "1": 3, "2": 3, "3": 3,
                "4": 4, "5": 3, "6": 2,
                "7": 2, "8": 3, "9": 2,
                "12": 4, "13": 3,
                "14": 4, "15": 3, "16": 4, "17": 3, "18": 4, "19": 3,
            },

            "settle_after_launch": {
                "1": 3, "2": 3, "3": 3,
                "4": 3, "5": 3, "6": 2,
                "7": 3, "8": 3, "9": 2,
                "12": 3, "13": 3,
                "14": 3, "15": 3, "16": 3, "17": 3, "18": 3, "19": 3,
            },

            "recover_forward": {
                "1": 2, "2": 2, "3": 2,
                "4": 3, "5": 3, "6": 2,
                "7": 3, "8": 3, "9": 2,
                "12": 3, "13": 3,
                "14": 3, "15": 3, "16": 4, "17": 4, "18": 3, "19": 3,
            },

            "recover_side": {
                "1": 3, "2": 3, "3": 3,
                "4": 4, "5": 3, "6": 2,
                "7": 4, "8": 3, "9": 2,
                "12": 3, "13": 3,
                "14": 3, "15": 3, "16": 3, "17": 3, "18": 3, "19": 3,
            },
        },

        "mutation": {
            "joint_mutation_rate": 0.12,
            "numeric_mutation_rate": 0.35,
            "small_numeric_sigma": 0.06,
            "allow_launch_mutation": False,
            "allow_transition_teacher_mutation": False,
            "mutable_numeric_paths": [
                "controller.foot_contact_z",
                "controller.min_step_forward",
                "controller.min_phase_frames",
                "controller.max_phase_frames",
                "controller.stable_hip_z",
                "controller.max_side_tilt",
                "controller.max_forward_drop",
                "arms.arm_drive_strength",
                "arms.max_pec_diff",
            ],
            "mutable_skills": [
                "step_left",
                "step_right",
                "push_left",
                "push_right",
                "settle_after_launch",
                "recover_forward",
                "recover_side",
            ],
        },
    }

    seed_path = Path(args.seed)
    champion_path = Path(args.champion)
    state_path = Path(args.state)

    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(json.dumps(agent, indent=2, ensure_ascii=False), encoding="utf-8")

    if not champion_path.exists():
        shutil.copyfile(seed_path, champion_path)
        print(f"Created champion from seed: {champion_path}")
    else:
        print(f"Champion already exists, not overwriting: {champion_path}")

    if not state_path.exists():
        state = {
            "stage": "teacher_single_step",
            "generation": 0,
            "champion_score": -10**9,
            "champion_valid_steps": 0,
            "champion_alternating_steps": 0,
            "champion_stable_frames": 0,
            "champion_pec_stability": 0,
            "teacher_imitation_weight": agent["controller"]["teacher_imitation_weight"],
            "physics_weight": agent["controller"]["physics_weight"],
            "notes": [
                "V1: le score vitesse est ignoré tant que alternating_steps < 2.",
                "Règle dure: ne pas remplacer un champion qui a acquis des pas alternés par un candidat sans alternance.",
            ],
        }
        state_path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Created curriculum state: {state_path}")

    print(f"Wrote seed: {seed_path}")


if __name__ == "__main__":
    main()
