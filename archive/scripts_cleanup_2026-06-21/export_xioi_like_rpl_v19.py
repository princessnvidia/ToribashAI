#!/usr/bin/env python3
"""
export_xioi_like_rpl_v19.py

Exporte des replays actions-only depuis les skills xioi-like V19.
But: tester visuellement si les pas extraits ressemblent enfin à de la marche.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
SKILLS_PATH = OUT_DIR / "xioi_like_step_skills_v19.json"
REPLAY_DIR = OUT_DIR
STEAM_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)

TURNFRAMES = 5
MATCHFRAMES = 1200
ENGAGE_Z = 0.0


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def by_category(skills: list[dict[str, Any]], cat: str) -> list[dict[str, Any]]:
    return [s for s in skills if s.get("category") == cat]


def skill_actions(skill: dict[str, Any]) -> list[list[list[int]]]:
    out = []
    for a in skill.get("actions", []):
        pairs = []
        for p in a.get("pairs", []):
            try:
                j, v = int(p[0]), int(p[1])
                if 0 <= j <= 19 and 1 <= v <= 4:
                    pairs.append([j, v])
            except Exception:
                pass
        out.append(pairs)
    return out


def compile_sequence(sequence: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    frame = 0
    actions = []
    last_pairs = None
    for label, skill in sequence:
        for pairs in skill_actions(skill):
            # Évite d'écrire deux fois exactement le même set si la source a des doublons.
            if pairs != last_pairs:
                actions.append({"frame": frame, "label": label, "skill_id": skill["id"], "pairs": pairs})
                last_pairs = pairs
            frame += TURNFRAMES
        frame += TURNFRAMES  # micro respiration entre skills
    return actions


def write_rpl(path: Path, fightname: str, actions: list[dict[str, Any]]) -> None:
    lines = []
    lines.append("#!/usr/bin/toribash")
    lines.append("#made with toribash-4.92")
    lines.append("#SCORE 0 0")
    lines.append("VERSION 12")
    lines.append(f"FIGHTNAME 0; {fightname}")
    lines.append("BOUT 0; ToribashAI")
    lines.append("BOUT 1; Uke")
    lines.append("AUTHOR 0; ToribashAI")
    lines.append(f"ENGAGE 0; 0.000000 -3.000000 {ENGAGE_Z:.6f} 0 0 0")
    lines.append(f"ENGAGE 1; 0.000000 0.000000 {ENGAGE_Z:.6f} 0 0 0")
    lines.append(f"NEWGAME 0;{MATCHFRAMES} 5 30 0 0 2 100 0 0 0 0 0 0 0 classic")
    lines.append("")
    for a in actions:
        lines.append(f"FRAME {int(a['frame'])};")
        lines.append(f"# {a['label']} skill={a['skill_id']}")
        for j, v in a["pairs"]:
            lines.append(f"JOINT 0; {j} {v}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def pick(pool: list[dict[str, Any]], idx: int) -> dict[str, Any]:
    if not pool:
        raise RuntimeError("Pool vide")
    return pool[idx % len(pool)]


def main() -> None:
    data = load_json(SKILLS_PATH)
    skills = data.get("skills", [])
    if not skills:
        raise RuntimeError("Aucun skill V19. Lance extract_xioi_like_steps_v19.py avant.")

    stand = by_category(skills, "stand_like")
    walk = by_category(skills, "walk_step_like")
    forward = by_category(skills, "forward_shift_like")
    recover = by_category(skills, "recover_like")

    print("Categories:", {c: len(by_category(skills, c)) for c in sorted(set(s.get('category') for s in skills))})

    tests: list[tuple[str, list[tuple[str, dict[str, Any]]]]] = []
    tests.append(("xioi_like_01_stand_walk_recover_v19", [
        ("stand", pick(stand or recover or walk, 0)),
        ("walk", pick(walk, 0)),
        ("recover", pick(recover or stand or walk, 0)),
    ]))
    tests.append(("xioi_like_02_walk_loop_v19", [
        ("stand", pick(stand or recover or walk, 1)),
        ("walk", pick(walk, 0)),
        ("walk", pick(walk, 1)),
        ("walk", pick(walk, 2)),
        ("walk", pick(walk, 3)),
        ("recover", pick(recover or stand or walk, 1)),
    ]))
    tests.append(("xioi_like_03_forward_walk_recover_v19", [
        ("stand", pick(stand or recover or walk, 2)),
        ("forward", pick(forward or walk, 0)),
        ("walk", pick(walk, 4)),
        ("walk", pick(walk, 5)),
        ("recover", pick(recover or stand or walk, 2)),
    ]))
    tests.append(("xioi_like_04_walk_only_v19", [
        ("walk", pick(walk, 6)),
        ("walk", pick(walk, 7)),
        ("walk", pick(walk, 8)),
        ("walk", pick(walk, 9)),
    ]))

    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    STEAM_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    written = []
    for name, seq in tests:
        actions = compile_sequence(seq)
        path = REPLAY_DIR / f"{name}.rpl"
        write_rpl(path, name, actions)
        shutil.copy2(path, STEAM_REPLAY_DIR / path.name)
        written.append(path)
        print("Wrote:", path, "actions", len(actions))
        print("Copied:", STEAM_REPLAY_DIR / path.name)

    print("Done. Open Setup > Replays and test xioi_like_*_v19.")


if __name__ == "__main__":
    main()
