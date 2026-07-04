#!/usr/bin/env python3
"""
export_real_skill_rpl_v17_2.py

V17.2 = export .rpl actions-only depuis les vrais skills humains V16.1,
mais sans mod custom et avec ENGAGE_Z = 0.0.

But:
  - oublier ToribashAI/toribashai_xioi_city_v1.tbm dans le replay
  - laisser Toribash charger classic
  - éviter POS/QAT/LINVEL/ANGVEL
  - ne garder que FRAME + JOINT

Entrée:
  generated_replays/parkour_real_replay_skills_v16_1.json

Sorties:
  generated_replays/*v17_2.rpl
  copie automatique vers le dossier replay Steam/Flatpak Toribash
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
SKILL_LIBRARY = OUT_DIR / "parkour_real_replay_skills_v16_1.json"

TORIBASH_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)

ENGAGE_Z = 0.0
TURNFRAMES = 5
MATCHFRAMES = 1200

# Classic volontairement, parce que le mod custom ne se charge pas depuis replay chez toi.
NEWGAME_MOD = "classic"

# Padding entre skills pour éviter les frames identiques ou trop compressées.
SKILL_GAP_FRAMES = 5


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def sanitize_pairs(pairs: list[Any]) -> list[tuple[int, int]]:
    clean: dict[int, int] = {}
    for pair in pairs or []:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        try:
            joint = int(pair[0])
            state = int(pair[1])
        except Exception:
            continue
        if 0 <= joint <= 19 and 1 <= state <= 4:
            clean[joint] = state
    return [(j, clean[j]) for j in sorted(clean)]


def group_skills(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for s in data.get("skills", []):
        cat = str(s.get("category", "unknown"))
        grouped.setdefault(cat, []).append(s)
    for cat in grouped:
        grouped[cat].sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)
    return grouped


def pick(grouped: dict[str, list[dict[str, Any]]], category: str, index: int = 0) -> dict[str, Any]:
    arr = grouped.get(category, [])
    if not arr:
        raise RuntimeError(f"Aucun skill dans la catégorie {category!r}")
    return arr[min(index, len(arr) - 1)]


def skill_actions(skill: dict[str, Any]) -> list[dict[str, Any]]:
    actions = skill.get("actions", [])
    out = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        pairs = sanitize_pairs(a.get("pairs", []))
        if not pairs:
            continue
        # dt peut être en turns/chunks dans certains exports, mais ici on le traite comme ordre relatif.
        out.append({"pairs": pairs})
    return out


def compile_sequence(sequence: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    frame = 0

    for label, skill in sequence:
        acts = skill_actions(skill)
        sid = skill.get("id", "?")
        cat = skill.get("category", "?")

        for i, action in enumerate(acts):
            frames.append({
                "frame": frame,
                "comment": f"skill {label}:{cat}:{sid}",
                "pairs": action["pairs"],
            })
            frame += TURNFRAMES

        frame += SKILL_GAP_FRAMES

    return frames


def write_rpl(path: Path, fightname: str, frames: list[dict[str, Any]]) -> None:
    lines: list[str] = []
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
    # Classic/no custom mod. Keep the mod token simple.
    lines.append(f"NEWGAME 0;{MATCHFRAMES} {TURNFRAMES} 30 0 0 2 100 0 0 0 0 0 0 0 {NEWGAME_MOD}")
    lines.append("")

    for item in frames:
        lines.append(f"FRAME {int(item['frame'])};")
        lines.append(f"# {item.get('comment', '')}")
        for joint, state in item.get("pairs", []):
            lines.append(f"JOINT 0; {joint} {state}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data = load_json(SKILL_LIBRARY)
    grouped = group_skills(data)

    print("Skill library:", SKILL_LIBRARY)
    print("Categories:", {k: len(v) for k, v in grouped.items()})
    print("ENGAGE_Z:", ENGAGE_Z)
    print("NEWGAME_MOD:", NEWGAME_MOD)

    # Des combinaisons simples, lisibles et comparables.
    recipes: list[tuple[str, list[tuple[str, dict[str, Any]]]]] = [
        (
            "parkour_real_skill_01_zero_classic_stand_forward_recover_v17_2",
            [
                ("stand0", pick(grouped, "stand", 0)),
                ("forward0", pick(grouped, "forward_impulse", 0)),
                ("recover0", pick(grouped, "recover", 0)),
            ],
        ),
        (
            "parkour_real_skill_02_zero_classic_walk_loop_v17_2",
            [
                ("stand0", pick(grouped, "stand", 0)),
                ("walk0", pick(grouped, "walk_step", 0)),
                ("walk1", pick(grouped, "walk_step", 1)),
                ("walk2", pick(grouped, "walk_step", 2)),
                ("recover0", pick(grouped, "recover", 0)),
            ],
        ),
        (
            "parkour_real_skill_03_zero_classic_forward_walk_recover_v17_2",
            [
                ("stand1", pick(grouped, "stand", 1)),
                ("forward1", pick(grouped, "forward_impulse", 3)),
                ("walk0", pick(grouped, "walk_step", 0)),
                ("recover1", pick(grouped, "recover", 1)),
            ],
        ),
        (
            "parkour_real_skill_04_zero_classic_forward_only_v17_2",
            [
                ("stand0", pick(grouped, "stand", 0)),
                ("forward0", pick(grouped, "forward_impulse", 0)),
            ],
        ),
        (
            "parkour_real_skill_05_zero_classic_walk_only_v17_2",
            [
                ("stand0", pick(grouped, "stand", 0)),
                ("walk0", pick(grouped, "walk_step", 0)),
                ("walk1", pick(grouped, "walk_step", 3)),
            ],
        ),
        (
            "parkour_real_skill_06_zero_classic_forward_acro_recover_v17_2",
            [
                ("stand0", pick(grouped, "stand", 0)),
                ("forward0", pick(grouped, "forward_impulse", 0)),
                ("acro0", pick(grouped, "acro", 0)),
                ("recover0", pick(grouped, "recover", 0)),
            ],
        ),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TORIBASH_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    # Nettoyage ciblé des anciens V17.2 côté projet + Steam pour éviter les doublons.
    for old in OUT_DIR.glob("*v17_2*.rpl"):
        old.unlink(missing_ok=True)
    for old in TORIBASH_REPLAY_DIR.glob("*v17_2*.rpl"):
        old.unlink(missing_ok=True)

    written = []
    for name, sequence in recipes:
        frames = compile_sequence(sequence)
        out_path = OUT_DIR / f"{name}.rpl"
        write_rpl(out_path, name, frames)
        steam_path = TORIBASH_REPLAY_DIR / out_path.name
        shutil.copy2(out_path, steam_path)
        written.append(out_path)
        print(f"wrote {out_path.name} frames={len(frames)} -> Steam")

    print("\nGenerated:")
    for p in written:
        print(" ", p)
    print("\nSteam replay dir:")
    print(" ", TORIBASH_REPLAY_DIR)
    print("\nHeader check:")
    first = written[0]
    print("\n".join(first.read_text(encoding="utf-8").splitlines()[:18]))


if __name__ == "__main__":
    main()
