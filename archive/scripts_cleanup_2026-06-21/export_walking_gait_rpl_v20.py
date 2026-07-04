#!/usr/bin/env python3
"""
export_walking_gait_rpl_v20.py

Exporte des RPL actions-only à partir des skills gait V20.
But: tester visuellement des séquences vraiment construites avec left_step/right_step.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
SKILLS_PATH = OUT_DIR / "parkour_walking_gait_skills_v20.json"
TORIBASH_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)

MATCHFRAMES = 1200
TURNFRAMES = 5
ENGAGE_Z = 0.0
MOD = "classic"


def load_skills() -> list[dict[str, Any]]:
    data = json.loads(SKILLS_PATH.read_text(encoding="utf-8"))
    return data["skills"]


def by_category(skills: list[dict[str, Any]], cat: str) -> list[dict[str, Any]]:
    arr = [s for s in skills if s.get("category") == cat]
    arr.sort(key=lambda s: float(s.get("score", 0)), reverse=True)
    return arr


def compile_sequence(sequence: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    actions = []
    t = 0
    for label, skill in sequence:
        skill_actions = skill.get("actions", [])
        if not skill_actions:
            continue
        base = int(skill_actions[0].get("dt", 0))
        for a in skill_actions:
            dt = int(a.get("dt", 0)) - base
            pairs = a.get("pairs", [])
            if not pairs:
                continue
            actions.append({
                "frame": t + max(0, dt),
                "pairs": [[int(j), int(v)] for j, v in pairs],
                "label": label,
                "skill_id": int(skill.get("id", -1)),
                "category": skill.get("category"),
            })
        last_dt = max([int(a.get("dt", 0)) - base for a in skill_actions] or [0])
        t += max(TURNFRAMES, last_dt + TURNFRAMES)
    # merge same-frame joint writes, last wins
    merged = {}
    meta = {}
    for a in actions:
        f = int(round(a["frame"] / TURNFRAMES) * TURNFRAMES)
        merged.setdefault(f, {})
        for j, v in a["pairs"]:
            merged[f][int(j)] = int(v)
        meta.setdefault(f, []).append(f"{a['label']}:{a['skill_id']}")
    out = []
    for f in sorted(merged):
        out.append({
            "frame": f,
            "pairs": [[j, merged[f][j]] for j in sorted(merged[f])],
            "comment": ",".join(meta.get(f, [])),
        })
    return out


def write_rpl(name: str, actions: list[dict[str, Any]]) -> Path:
    path = OUT_DIR / f"{name}.rpl"
    lines = [
        "#!/usr/bin/toribash",
        "#made with toribash-4.92",
        "#SCORE 0 0",
        "VERSION 12",
        f"FIGHTNAME 0; {name}",
        "BOUT 0; ToribashAI",
        "BOUT 1; Uke",
        "AUTHOR 0; ToribashAI",
        f"ENGAGE 0; 0.000000 -3.000000 {ENGAGE_Z:.6f} 0 0 0",
        f"ENGAGE 1; 0.000000 0.000000 {ENGAGE_Z:.6f} 0 0 0",
        f"NEWGAME 0;{MATCHFRAMES} {TURNFRAMES} 30 0 0 2 100 0 0 0 0 0 0 0 {MOD}",
        "",
    ]
    for a in actions:
        lines.append(f"FRAME {int(a['frame'])};")
        if a.get("comment"):
            lines.append(f"# {a['comment']}")
        for j, v in a.get("pairs", []):
            lines.append(f"JOINT 0; {int(j)} {int(v)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def pick(arr: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    if not arr:
        raise RuntimeError("Catégorie vide")
    return arr[idx % len(arr)]


def main() -> None:
    skills = load_skills()
    stand = by_category(skills, "stand")
    lean = by_category(skills, "lean_forward")
    left = by_category(skills, "left_step")
    right = by_category(skills, "right_step")
    recover = by_category(skills, "recover_upright")

    print("Counts:", {"stand": len(stand), "lean": len(lean), "left": len(left), "right": len(right), "recover": len(recover)})
    if not left or not right:
        raise RuntimeError("Il manque left_step ou right_step. Relance extraction ou baisse les seuils.")

    sequences = []
    sequences.append((
        "parkour_gait_v20_01_left_right_loop",
        [("stand", pick(stand, 0)), ("lean", pick(lean, 0) if lean else pick(stand, 1)),
         ("left", pick(left, 0)), ("right", pick(right, 0)),
         ("left", pick(left, 1)), ("right", pick(right, 1)),
         ("recover", pick(recover, 0) if recover else pick(stand, 2))]
    ))
    sequences.append((
        "parkour_gait_v20_02_right_left_loop",
        [("stand", pick(stand, 1)), ("lean", pick(lean, 1) if len(lean) > 1 else pick(stand, 0)),
         ("right", pick(right, 2)), ("left", pick(left, 2)),
         ("right", pick(right, 3)), ("left", pick(left, 3)),
         ("recover", pick(recover, 1) if len(recover) > 1 else pick(stand, 3))]
    ))
    sequences.append((
        "parkour_gait_v20_03_steps_only",
        [("left", pick(left, 4)), ("right", pick(right, 4)), ("left", pick(left, 5)), ("right", pick(right, 5)),
         ("left", pick(left, 6)), ("right", pick(right, 6))]
    ))
    sequences.append((
        "parkour_gait_v20_04_soft_start_steps",
        [("stand", pick(stand, 2)), ("stand", pick(stand, 3)),
         ("left", pick(left, 7)), ("right", pick(right, 7)),
         ("recover", pick(recover, 2) if len(recover) > 2 else pick(stand, 4))]
    ))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TORIBASH_REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for name, seq in sequences:
        actions = compile_sequence(seq)
        p = write_rpl(name, actions)
        shutil.copy2(p, TORIBASH_REPLAY_DIR / p.name)
        written.append(p)
        print("Wrote:", p, "actions", len(actions))
        print("Copied:", TORIBASH_REPLAY_DIR / p.name)

    print("\nDone. Teste les replays v20 dans Toribash.")


if __name__ == "__main__":
    main()
