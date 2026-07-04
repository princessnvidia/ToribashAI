#!/usr/bin/env python3
"""
export_early_walking_axis_rpl_v22.py

Exporte des RPL actions-only depuis early_walking_axis_skills_v22.json.
Le but est de tester visuellement si les skills V22 ressemblent vraiment à des pas.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
SKILLS_PATH = OUT_DIR / "early_walking_axis_skills_v22.json"
REPLAY_DIR = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"

ENGAGE_Z = 0.0
TURNFRAMES = 5
MATCHFRAMES = 1200
MOD_NAME = "classic"


def load_skills() -> list[dict[str, Any]]:
    data = json.loads(SKILLS_PATH.read_text(encoding="utf-8"))
    return data.get("skills", [])


def select(skills: list[dict[str, Any]], prefix: str, n: int) -> list[dict[str, Any]]:
    arr = [s for s in skills if str(s.get("category", "")).startswith(prefix)]
    arr.sort(key=lambda s: float(s.get("score", 0)), reverse=True)
    return arr[:n]


def write_rpl(name: str, sequence: list[dict[str, Any]]) -> Path:
    lines: list[str] = []
    lines += [
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
        f"NEWGAME 0;{MATCHFRAMES} {TURNFRAMES} 30 0 0 2 100 0 0 0 0 0 0 0 {MOD_NAME}",
        "",
    ]

    frame_cursor = 0
    for skill in sequence:
        actions = skill.get("actions", [])
        for action in actions:
            dt = int(action.get("dt", 0))
            frame = frame_cursor + dt
            lines.append(f"FRAME {frame};")
            lines.append(f"# skill {skill.get('id')} {skill.get('category')} {skill.get('replay')} {skill.get('start_frame')}-{skill.get('end_frame')}")
            for pair in action.get("pairs", []):
                if isinstance(pair, list) and len(pair) >= 2:
                    lines.append(f"JOINT 0; {int(pair[0])} {int(pair[1])}")
            lines.append("")
        # Ajoute un peu d'espace entre skills, basé sur durée du skill.
        max_dt = max([int(a.get("dt", 0)) for a in actions] + [0])
        frame_cursor += max(15, max_dt + TURNFRAMES)

    path = OUT_DIR / f"{name}.rpl"
    path.write_text("\n".join(lines), encoding="utf-8")
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, REPLAY_DIR / path.name)
    return path


def main() -> None:
    skills = load_skills()
    print("skills:", len(skills))
    if not skills:
        raise RuntimeError("Aucun skill V22. Lance extract_early_walking_axis_v22.py d'abord.")

    left = select(skills, "step_left", 12)
    right = select(skills, "step_right", 12)
    alltop = sorted(skills, key=lambda s: float(s.get("score", 0)), reverse=True)[:20]

    print("left:", len(left), "right:", len(right))
    sequences: list[tuple[str, list[dict[str, Any]]]] = []

    if left and right:
        seq = []
        for i in range(min(5, len(left), len(right))):
            seq.append(left[i])
            seq.append(right[i])
        sequences.append(("walking_axis_v22_01_left_right_alternate", seq))

    if alltop:
        sequences.append(("walking_axis_v22_02_top10", alltop[:10]))
        sequences.append(("walking_axis_v22_03_top5_loop", (alltop[:5] * 2)))

    if left:
        sequences.append(("walking_axis_v22_04_left_only_test", left[:8]))
    if right:
        sequences.append(("walking_axis_v22_05_right_only_test", right[:8]))

    # Séquence plus lente : répète les meilleurs micro/forward modérés.
    moderate = [s for s in skills if 0.15 <= float(s.get("forward", 0)) <= 2.2]
    moderate.sort(key=lambda s: float(s.get("score", 0)), reverse=True)
    if moderate:
        sequences.append(("walking_axis_v22_06_moderate_steps", moderate[:12]))

    written = []
    for name, seq in sequences:
        p = write_rpl(name, seq)
        written.append(p)
        print("wrote:", p)

    print("\nCopied to:", REPLAY_DIR)
    print("Generated:", len(written))


if __name__ == "__main__":
    main()
