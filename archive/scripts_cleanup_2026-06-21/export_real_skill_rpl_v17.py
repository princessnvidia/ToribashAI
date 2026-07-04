#!/usr/bin/env python3
"""
export_real_skill_rpl_v17.py

V17 = export RPL tests from real human replay skills extracted by V16.1.

Input:
  generated_replays/parkour_real_replay_skills_v16_1.json

Output project:
  generated_replays/parkour_real_skill_test_01_stand_forward_recover_v17.rpl
  generated_replays/parkour_real_skill_test_02_stand_walk_recover_v17.rpl
  generated_replays/parkour_real_skill_test_03_stand_forward_walk_recover_v17.rpl
  generated_replays/parkour_real_skill_test_04_walk_loop_v17.rpl
  generated_replays/parkour_real_skill_test_05_forward_only_v17.rpl

Output Toribash replay folder:
  same files copied to Steam Toribash replay directory.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
SKILLS_PATH = OUT_DIR / "parkour_real_replay_skills_v16_1.json"

TORIBASH_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)

MOD_NAME = "ToribashAI/toribashai_xioi_city_v1.tbm"
TURNFRAMES = 5
MATCH_FRAMES = 1200

# On garde une pause entre skills pour laisser la physique répondre.
GAP_TURNS = 1

# Limite les skills très longs pour que les tests restent lisibles.
MAX_ACTIONS_PER_SKILL = 18


def load_skills() -> dict[str, Any]:
    if not SKILLS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {SKILLS_PATH}\nRun: python3 scripts/extract_real_replay_skills_v16_1.py"
        )
    return json.loads(SKILLS_PATH.read_text(encoding="utf-8"))


def by_category(data: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    cats: dict[str, list[dict[str, Any]]] = {}
    for s in data.get("skills", []):
        cats.setdefault(str(s.get("category", "unknown")), []).append(s)
    for items in cats.values():
        items.sort(key=lambda s: float(s.get("score", 0.0)), reverse=True)
    return cats


def choose(cats: dict[str, list[dict[str, Any]]], category: str, index: int = 0) -> dict[str, Any]:
    items = cats.get(category, [])
    if not items:
        raise RuntimeError(f"No skills in category {category!r}")
    return items[index % len(items)]


def clean_pairs(pairs: Any) -> list[list[int]]:
    out: list[list[int]] = []
    seen: set[int] = set()
    if not isinstance(pairs, list):
        return out
    for pair in pairs:
        if not isinstance(pair, list | tuple) or len(pair) < 2:
            continue
        try:
            j = int(pair[0])
            v = int(pair[1])
        except Exception:
            continue
        if 0 <= j <= 19 and 1 <= v <= 4:
            seen.add(j)
            out.append([j, v])
    # si même joint présent plusieurs fois, le dernier gagne
    merged: dict[int, int] = {}
    for j, v in out:
        merged[j] = v
    return [[j, merged[j]] for j in sorted(merged)]


def skill_actions(skill: dict[str, Any]) -> list[dict[str, Any]]:
    actions = skill.get("actions", [])
    if not isinstance(actions, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for a in actions[:MAX_ACTIONS_PER_SKILL]:
        if not isinstance(a, dict):
            continue
        pairs = clean_pairs(a.get("pairs", []))
        if pairs:
            cleaned.append({"pairs": pairs})
    return cleaned


def compile_sequence(sequence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    turn = 0

    # frame 0 : pose neutre/hold-ish très simple pour partir propre.
    frames.append({
        "frame": 0,
        "pairs": [[2, 3], [3, 3], [16, 3], [17, 3], [18, 3], [19, 3]],
        "skill_id": "warmup",
        "category": "warmup",
    })
    turn += 2

    for skill in sequence:
        acts = skill_actions(skill)
        for a in acts:
            frames.append({
                "frame": turn * TURNFRAMES,
                "pairs": a["pairs"],
                "skill_id": int(skill.get("id", -1)),
                "category": str(skill.get("category", "unknown")),
                "source_replay": str(skill.get("replay", "")),
                "source_frames": f"{skill.get('start_frame')}–{skill.get('end_frame')}",
            })
            turn += 1
        turn += GAP_TURNS

    return frames


def write_rpl(path: Path, name: str, frames: list[dict[str, Any]]) -> None:
    lines: list[str] = []
    lines.append("#!/usr/bin/toribash")
    lines.append("#made with toribash-4.92")
    lines.append("#SCORE 0 0")
    lines.append("#WIN 0 ToribashAI uke")
    lines.append("VERSION 12")
    lines.append(f"FIGHTNAME 0; {name}")
    lines.append("BOUT 0; ToribashAI")
    lines.append("BOUT 1; uke")
    lines.append("AUTHOR 0; ToribashAI")
    lines.append("ENGAGE 0; 0.000000 -3.000000 5.400000 0 0 0")
    lines.append("ENGAGE 1; 0.000000 0.000000 5.400000 0 0 0")
    lines.append(f"NEWGAME 0;{MATCH_FRAMES} {TURNFRAMES} 30 0 0 0 0 0 0 0 0 0 0 {MOD_NAME}")
    lines.append("")

    for item in frames:
        frame = int(item["frame"])
        lines.append(f"FRAME {frame};")
        lines.append(f"# skill={item.get('skill_id')} category={item.get('category')} src={item.get('source_frames', '')}")
        for j, v in item["pairs"]:
            lines.append(f"JOINT 0; {int(j)} {int(v)}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def describe_sequence(seq: list[dict[str, Any]]) -> str:
    return " -> ".join(f"{s.get('category')}#{s.get('id')}" for s in seq)


def main() -> None:
    data = load_skills()
    cats = by_category(data)

    print("Loaded:", SKILLS_PATH)
    print("Categories:", {k: len(v) for k, v in sorted(cats.items())})

    # On évite les top forward/walk trop bas en tête si possible.
    # Mais on garde plusieurs variantes pour voir visuellement.
    stand0 = choose(cats, "stand", 0)
    stand1 = choose(cats, "stand", 1)
    forward0 = choose(cats, "forward_impulse", 0)
    forward3 = choose(cats, "forward_impulse", 3)
    walk0 = choose(cats, "walk_step", 0)
    walk2 = choose(cats, "walk_step", 2)
    walk5 = choose(cats, "walk_step", 5)
    recover0 = choose(cats, "recover", 0)
    recover2 = choose(cats, "recover", 2)
    acro0 = choose(cats, "acro", 0)

    tests: list[tuple[str, list[dict[str, Any]]]] = [
        (
            "parkour_real_skill_test_01_stand_forward_recover_v17.rpl",
            [stand0, stand0, forward0, recover0, stand1],
        ),
        (
            "parkour_real_skill_test_02_stand_walk_recover_v17.rpl",
            [stand0, walk0, walk2, walk5, recover0, stand1],
        ),
        (
            "parkour_real_skill_test_03_stand_forward_walk_recover_v17.rpl",
            [stand0, forward3, walk0, walk2, recover2, stand1],
        ),
        (
            "parkour_real_skill_test_04_walk_loop_v17.rpl",
            [stand0, walk0, recover2, walk2, recover0, walk5, stand1],
        ),
        (
            "parkour_real_skill_test_05_forward_only_v17.rpl",
            [stand0, forward0, forward3, recover0],
        ),
        (
            "parkour_real_skill_test_06_stand_forward_acro_recover_v17.rpl",
            [stand0, forward3, acro0, recover0, stand1],
        ),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TORIBASH_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    for filename, seq in tests:
        frames = compile_sequence(seq)
        out_path = OUT_DIR / filename
        write_rpl(out_path, filename.replace(".rpl", ""), frames)
        steam_path = TORIBASH_REPLAY_DIR / filename
        shutil.copy2(out_path, steam_path)
        manifest.append({
            "file": str(out_path),
            "steam_file": str(steam_path),
            "sequence": [
                {
                    "id": int(s.get("id", -1)),
                    "category": str(s.get("category", "unknown")),
                    "score": float(s.get("score", 0.0)),
                    "dx": float(s.get("dx", 0.0)),
                    "dy": float(s.get("dy", 0.0)),
                    "head": float(s.get("head_z", s.get("head", 0.0))),
                    "frames": f"{s.get('start_frame')}–{s.get('end_frame')}",
                    "replay": str(s.get("replay", "")),
                }
                for s in seq
            ],
            "compiled_frames": len(frames),
        })
        print("\nWrote:", out_path.name)
        print("  ", describe_sequence(seq))
        print("  copied to:", steam_path)

    manifest_path = OUT_DIR / "parkour_real_skill_tests_v17_manifest.json"
    manifest_path.write_text(json.dumps({"version": 17, "tests": manifest}, indent=2), encoding="utf-8")
    print("\nManifest:", manifest_path)
    print("\nOpen Toribash > Setup > Replays and test the parkour_real_skill_test_*_v17 files.")


if __name__ == "__main__":
    main()
