#!/usr/bin/env python3
"""
export_locomotion_rpl_v18.py

Génère des replays actions-only depuis la bibliothèque locomotion V18.
But: tester marche / impulsion contrôlée sans POS/QAT et sans mod custom.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
SKILLS_PATH = OUT_DIR / "parkour_locomotion_skills_v18.json"
REPLAY_DIR = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"

TURNFRAMES = 5
MATCHFRAMES = 1200
ENGAGE_Z = 0.0
MOD = "classic"


def load_skills() -> dict[str, list[dict[str, Any]]]:
    data = json.loads(SKILLS_PATH.read_text(encoding="utf-8"))
    by = {}
    for s in data.get("skills", []):
        by.setdefault(s.get("category", "unknown"), []).append(s)
    for v in by.values():
        v.sort(key=lambda s: s.get("score", 0), reverse=True)
    return by


def pick(by: dict[str, list[dict[str, Any]]], cat: str, idx: int = 0) -> dict[str, Any]:
    items = by.get(cat) or []
    if not items:
        raise RuntimeError(f"Pas de skill category={cat}")
    return items[idx % len(items)]


def flatten(seq: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    actions = []
    t = 0
    last_frame_written = -1
    for label, skill in seq:
        skill_actions = skill.get("actions", [])
        if not skill_actions:
            continue
        base_dt = int(skill_actions[0].get("dt", 0))
        max_local = 0
        for a in skill_actions:
            dt = int(a.get("dt", 0)) - base_dt
            # compresse un peu les gros gaps issus des replays longs
            local_frame = max(0, min(dt, 45))
            frame = t + local_frame
            if frame <= last_frame_written:
                frame = last_frame_written + TURNFRAMES
            pairs = []
            for pair in a.get("pairs", []):
                try:
                    j = int(pair[0]); v = int(pair[1])
                    if 0 <= j <= 19 and 1 <= v <= 4:
                        pairs.append([j, v])
                except Exception:
                    pass
            if pairs:
                actions.append({"frame": frame, "pairs": pairs, "label": label, "skill_id": skill.get("id")})
                last_frame_written = frame
                max_local = max(max_local, local_frame)
        t += max(25, max_local + 10)
    return actions


def write_rpl(name: str, seq: list[tuple[str, dict[str, Any]]]) -> Path:
    actions = flatten(seq)
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
        lines.append(f"# {a['label']} skill={a.get('skill_id')}")
        # combine duplicate joints in same frame
        clean = {}
        for j, v in a["pairs"]:
            clean[int(j)] = int(v)
        for j in sorted(clean):
            lines.append(f"JOINT 0; {j} {clean[j]}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    if not SKILLS_PATH.exists():
        raise FileNotFoundError(f"Lance d'abord extract_locomotion_skills_v18.py: {SKILLS_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    by = load_skills()
    print("Categories:", {k: len(v) for k, v in by.items()})

    recipes = [
        (
            "parkour_locomotion_01_stand_lean_walk_recover_v18",
            [
                ("stand", pick(by, "stand", 0)),
                ("lean", pick(by, "lean_forward", 0)),
                ("walk", pick(by, "walk_step", 0)),
                ("recover", pick(by, "recover_upright", 0)),
            ],
        ),
        (
            "parkour_locomotion_02_walk_loop_v18",
            [
                ("stand", pick(by, "stand", 1)),
                ("walk", pick(by, "walk_step", 1)),
                ("walk", pick(by, "walk_step", 2)),
                ("walk", pick(by, "walk_step", 3)),
                ("recover", pick(by, "recover_upright", 1)),
            ],
        ),
        (
            "parkour_locomotion_03_step_lr_v18",
            [
                ("stand", pick(by, "stand", 2)),
                ("left", pick(by, "step_left", 0)),
                ("right", pick(by, "step_right", 0)),
                ("left", pick(by, "step_left", 1)),
                ("right", pick(by, "step_right", 1)),
                ("recover", pick(by, "recover_upright", 2)),
            ],
        ),
        (
            "parkour_locomotion_04_lean_only_v18",
            [
                ("stand", pick(by, "stand", 0)),
                ("lean", pick(by, "lean_forward", 0)),
                ("lean", pick(by, "lean_forward", 1)),
                ("lean", pick(by, "lean_forward", 2)),
                ("recover", pick(by, "recover_upright", 0)),
            ],
        ),
        (
            "parkour_locomotion_05_best_walk_only_v18",
            [
                ("stand", pick(by, "stand", 0)),
                ("walk", pick(by, "walk_step", 0)),
                ("walk", pick(by, "walk_step", 4)),
                ("walk", pick(by, "walk_step", 8)),
                ("recover", pick(by, "recover_upright", 3)),
            ],
        ),
    ]

    made = []
    # remove old v18 locomotion replays first
    for p in REPLAY_DIR.glob("*locomotion*v18*.rpl"):
        p.unlink(missing_ok=True)
    for name, seq in recipes:
        p = write_rpl(name, seq)
        dst = REPLAY_DIR / p.name
        shutil.copy2(p, dst)
        made.append(dst)
        print("wrote:", p)
        print("copied:", dst)
    print("\nDone. Teste les replays parkour_locomotion_*_v18 dans Toribash.")


if __name__ == "__main__":
    main()
