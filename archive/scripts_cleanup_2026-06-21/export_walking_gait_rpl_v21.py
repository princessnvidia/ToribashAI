#!/usr/bin/env python3
"""
export_walking_gait_rpl_v21.py

Exporte des replays .rpl actions-only depuis walking_gait_strict_skills_v21.json.
But: tester visuellement uniquement les skills de marche stricte V21.

Sorties copiées dans:
  generated_replays/*.rpl
  Toribash/replay/*.rpl
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
SKILLS_PATH = OUT_DIR / "walking_gait_strict_skills_v21.json"

TORIBASH_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)

TURNFRAMES = 5
MATCHFRAMES = 1200
ENGAGE_Z = 0.0
MOD_NAME = "classic"


def load_skills() -> list[dict[str, Any]]:
    data = json.loads(SKILLS_PATH.read_text(encoding="utf-8"))
    return data.get("skills", [])


def select(skills, category=None, count=1, offset=0):
    arr = [s for s in skills if category is None or s.get("category") == category]
    arr.sort(key=lambda s: s.get("score", 0), reverse=True)
    if not arr:
        return []
    out = []
    for i in range(count):
        out.append(arr[(offset + i) % len(arr)])
    return out


def normalize_skill_actions(skill: dict[str, Any], start_frame: int) -> tuple[list[dict[str, Any]], int]:
    actions = []
    last_frame = start_frame
    for a in skill.get("actions", []):
        dt = int(a.get("dt", 0))
        # Les dt des replays humains peuvent être 10/20/etc. On compresse doucement en turns de 5.
        frame = start_frame + max(0, round(dt / 5) * TURNFRAMES)
        pairs = []
        clean = {}
        for pair in a.get("pairs", []):
            if isinstance(pair, list) and len(pair) >= 2:
                try:
                    j = int(pair[0]); v = int(pair[1])
                except Exception:
                    continue
                if 0 <= j < 20 and 1 <= v <= 4:
                    clean[j] = v
        if clean:
            pairs = [[j, clean[j]] for j in sorted(clean)]
            actions.append({
                "frame": frame,
                "pairs": pairs,
                "skill_id": skill.get("id"),
                "category": skill.get("category"),
            })
            last_frame = max(last_frame, frame)
    return actions, last_frame + TURNFRAMES


def compile_sequence(sequence: list[dict[str, Any]], gap: int = 5) -> list[dict[str, Any]]:
    all_actions = []
    cursor = 0
    for skill in sequence:
        actions, next_cursor = normalize_skill_actions(skill, cursor)
        all_actions.extend(actions)
        cursor = next_cursor + gap

    # Merge same frame.
    by_frame = {}
    meta = {}
    for a in all_actions:
        f = int(a["frame"])
        by_frame.setdefault(f, {})
        for j, v in a["pairs"]:
            by_frame[f][int(j)] = int(v)
        meta.setdefault(f, []).append(f"{a.get('category')}:{a.get('skill_id')}")

    compiled = []
    for f in sorted(by_frame):
        compiled.append({
            "frame": f,
            "pairs": [[j, by_frame[f][j]] for j in sorted(by_frame[f])],
            "comment": ",".join(meta.get(f, [])),
        })
    return compiled


def rpl_text(name: str, actions: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("#!/usr/bin/toribash")
    lines.append("#made with toribash-4.92")
    lines.append("#SCORE 0 0")
    lines.append("VERSION 12")
    lines.append(f"FIGHTNAME 0; {name}")
    lines.append("BOUT 0; ToribashAI")
    lines.append("BOUT 1; Uke")
    lines.append("AUTHOR 0; ToribashAI")
    lines.append(f"ENGAGE 0; 0.000000 -3.000000 {ENGAGE_Z:.6f} 0 0 0")
    lines.append(f"ENGAGE 1; 0.000000 0.000000 {ENGAGE_Z:.6f} 0 0 0")
    lines.append(f"NEWGAME 0;{MATCHFRAMES} {TURNFRAMES} 30 0 0 2 100 0 0 0 0 0 0 0 {MOD_NAME}")
    lines.append("")

    for a in actions:
        lines.append(f"FRAME {int(a['frame'])};")
        if a.get("comment"):
            lines.append(f"# {a['comment']}")
        for j, v in a.get("pairs", []):
            lines.append(f"JOINT 0; {int(j)} {int(v)}")
        lines.append("")

    return "\n".join(lines) + "\n"


def save_replay(name: str, sequence: list[dict[str, Any]]):
    actions = compile_sequence(sequence)
    path = OUT_DIR / f"{name}.rpl"
    path.write_text(rpl_text(name, actions), encoding="utf-8")
    TORIBASH_REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, TORIBASH_REPLAY_DIR / path.name)
    print("saved:", path)
    print("copied:", TORIBASH_REPLAY_DIR / path.name)
    return path


def main() -> None:
    if not SKILLS_PATH.exists():
        raise FileNotFoundError(f"Missing {SKILLS_PATH}. Run extract_walking_gait_skills_v21.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    skills = load_skills()
    print("skills:", len(skills))
    cats = sorted(set(s.get("category") for s in skills))
    print("categories:", cats)

    step_left = select(skills, "step_left", 8)
    step_right = select(skills, "step_right", 8)
    upright = select(skills, "upright_step", 8)
    micro = select(skills, "micro_step", 8)

    # Fallback si les catégories sont rares.
    all_best = select(skills, None, 16)
    if not step_left: step_left = all_best[::2]
    if not step_right: step_right = all_best[1::2]
    if not upright: upright = all_best[:4]
    if not micro: micro = all_best[-4:]

    sequences = []
    sequences.append(("walking_gait_v21_01_upright_micro_steps", upright[:2] + micro[:4] + upright[2:4]))
    sequences.append(("walking_gait_v21_02_left_right_alternate", [step_left[0], step_right[0], step_left[1], step_right[1], step_left[2], step_right[2]]))
    sequences.append(("walking_gait_v21_03_right_left_alternate", [step_right[0], step_left[0], step_right[1], step_left[1], step_right[2], step_left[2]]))
    sequences.append(("walking_gait_v21_04_best_stable_loop", all_best[:8]))
    sequences.append(("walking_gait_v21_05_slow_gait_mix", upright[:1] + [step_left[0], micro[0], step_right[0], micro[1], step_left[1], step_right[1]]))
    sequences.append(("walking_gait_v21_06_top_single_steps", all_best[:5]))

    for name, seq in sequences:
        seq = [s for s in seq if s]
        if seq:
            save_replay(name, seq)

    print("\nDone. Open Setup > Replays and test walking_gait_v21_*.rpl")


if __name__ == "__main__":
    main()
