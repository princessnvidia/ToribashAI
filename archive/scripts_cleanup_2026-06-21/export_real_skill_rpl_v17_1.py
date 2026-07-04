#!/usr/bin/env python3
"""
export_real_skill_rpl_v17_1.py

V17.1 = export RPL actions-only depuis les vrais skills humains V16.1.

But du fix:
  - ne PAS écrire POS/QAT/LINVEL/ANGVEL des replays sources
  - laisser Toribash recalculer la physique depuis ENGAGE au sol
  - écrire uniquement NEWGAME + ENGAGE + FRAME + JOINT

Entrée:
  generated_replays/parkour_real_replay_skills_v16_1.json

Sorties:
  generated_replays/*.rpl
  + copie directe dans le dossier replay Steam/Flatpak Toribash
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
MATCHFRAMES = 1200
TURNFRAMES = 5

# Engage au sol pour notre mod plat. On évite les hauteurs des replays source.
ENGAGE_TORI = "0.000000 -3.000000 5.400000 0 0 0"
ENGAGE_UKE = "0.000000 0.000000 5.400000 0 0 0"


def load_skill_library() -> list[dict[str, Any]]:
    if not SKILLS_PATH.exists():
        raise FileNotFoundError(SKILLS_PATH)
    data = json.loads(SKILLS_PATH.read_text(encoding="utf-8"))
    skills = data.get("skills") or []
    if not skills:
        raise RuntimeError("Skill library vide")
    return skills


def by_category(skills: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for s in skills:
        cat = str(s.get("category", "unknown"))
        out.setdefault(cat, []).append(s)
    for cat in out:
        out[cat].sort(key=lambda s: float(s.get("score", 0)), reverse=True)
    return out


def skill_actions(skill: dict[str, Any]) -> list[dict[str, Any]]:
    actions = skill.get("actions") or []
    cleaned = []
    for idx, a in enumerate(actions):
        pairs = a.get("pairs") or []
        clean_pairs = []
        for pair in pairs:
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            try:
                j = int(pair[0])
                v = int(pair[1])
            except Exception:
                continue
            if 0 <= j <= 19 and 1 <= v <= 4:
                clean_pairs.append([j, v])
        if clean_pairs:
            dt = int(a.get("dt", idx))
            cleaned.append({"dt": dt, "pairs": clean_pairs})
    return cleaned


def append_skill(compiled: list[dict[str, Any]], skill: dict[str, Any], label: str, start_turn: int) -> int:
    actions = skill_actions(skill)
    if not actions:
        return start_turn

    # Rebase dt localement pour éviter de reprendre les frames absolues du replay source.
    min_dt = min(int(a.get("dt", 0)) for a in actions)
    for a in actions:
        turn = start_turn + (int(a.get("dt", 0)) - min_dt)
        compiled.append({
            "turn": turn,
            "frame": turn * TURNFRAMES,
            "pairs": a["pairs"],
            "skill_id": skill.get("id"),
            "category": skill.get("category"),
            "label": label,
        })

    max_dt = max(int(a.get("dt", 0)) for a in actions) - min_dt
    # Petite respiration entre skills.
    return start_turn + max_dt + 2


def compile_sequence(sequence: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    compiled: list[dict[str, Any]] = []
    turn = 0
    for label, skill in sequence:
        turn = append_skill(compiled, skill, label, turn)
    compiled.sort(key=lambda x: (x["frame"], str(x.get("label", ""))))

    # Fusion si plusieurs actions tombent sur la même frame : dernière valeur par joint.
    by_frame: dict[int, dict[int, int]] = {}
    meta: dict[int, list[str]] = {}
    for a in compiled:
        frame = int(a["frame"])
        by_frame.setdefault(frame, {})
        meta.setdefault(frame, [])
        meta[frame].append(f"{a.get('label')}:{a.get('skill_id')}")
        for j, v in a["pairs"]:
            by_frame[frame][int(j)] = int(v)

    out = []
    for frame in sorted(by_frame):
        pairs = [[j, by_frame[frame][j]] for j in sorted(by_frame[frame])]
        out.append({"frame": frame, "pairs": pairs, "meta": "+".join(meta[frame])})
    return out


def rpl_text(name: str, actions: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append("#!/usr/bin/toribash")
    lines.append("#made with toribash-4.92")
    lines.append("#SCORE 0 0")
    lines.append("VERSION 12")
    lines.append(f"FIGHTNAME 0; {name}")
    lines.append("BOUT 0; ToribashAI")
    lines.append("BOUT 1; Uke")
    lines.append("AUTHOR 0; ToribashAI")
    lines.append(f"ENGAGE 0; {ENGAGE_TORI}")
    lines.append(f"ENGAGE 1; {ENGAGE_UKE}")
    lines.append(f"NEWGAME 0;{MATCHFRAMES} {TURNFRAMES} 30 0 0 2 100 0 0 0 0 0 0 0 {MOD_NAME}")
    lines.append("")

    for a in actions:
        frame = int(a["frame"])
        lines.append(f"FRAME {frame};")
        lines.append(f"# skill {a.get('meta', '')}")
        for j, v in a["pairs"]:
            lines.append(f"JOINT 0; {j} {v}")
        lines.append("")

    # Aucun POS/QAT/LINVEL/ANGVEL ici.
    return "\n".join(lines) + "\n"


def pick(cat: dict[str, list[dict[str, Any]]], category: str, index: int = 0) -> dict[str, Any]:
    items = cat.get(category) or []
    if not items:
        raise RuntimeError(f"Aucun skill catégorie {category}")
    return items[min(index, len(items) - 1)]


def main() -> None:
    skills = load_skill_library()
    cat = by_category(skills)

    print("Categories:", {k: len(v) for k, v in cat.items()})

    stand0 = pick(cat, "stand", 0)
    stand1 = pick(cat, "stand", 1)
    recover0 = pick(cat, "recover", 0)
    recover1 = pick(cat, "recover", 1)
    forward0 = pick(cat, "forward_impulse", 0)
    forward1 = pick(cat, "forward_impulse", 3)
    walk0 = pick(cat, "walk_step", 0)
    walk1 = pick(cat, "walk_step", 2)
    walk2 = pick(cat, "walk_step", 4)
    acro0 = pick(cat, "acro", 0)

    tests: list[tuple[str, list[tuple[str, dict[str, Any]]]]] = [
        (
            "parkour_real_skill_01_actions_only_stand_forward_recover_v17_1.rpl",
            [("stand0", stand0), ("forward0", forward0), ("recover0", recover0)],
        ),
        (
            "parkour_real_skill_02_actions_only_walk_loop_v17_1.rpl",
            [("stand0", stand0), ("walk0", walk0), ("walk1", walk1), ("walk2", walk2), ("recover0", recover0)],
        ),
        (
            "parkour_real_skill_03_actions_only_forward_walk_recover_v17_1.rpl",
            [("stand1", stand1), ("forward1", forward1), ("walk0", walk0), ("walk1", walk1), ("recover1", recover1)],
        ),
        (
            "parkour_real_skill_04_actions_only_forward_only_v17_1.rpl",
            [("stand0", stand0), ("forward0", forward0)],
        ),
        (
            "parkour_real_skill_05_actions_only_walk_only_v17_1.rpl",
            [("stand0", stand0), ("walk0", walk0), ("walk1", walk1), ("walk2", walk2)],
        ),
        (
            "parkour_real_skill_06_actions_only_forward_acro_recover_v17_1.rpl",
            [("stand0", stand0), ("forward0", forward0), ("acro0", acro0), ("recover0", recover0)],
        ),
    ]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TORIBASH_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    for filename, seq in tests:
        actions = compile_sequence(seq)
        text = rpl_text(filename.replace(".rpl", ""), actions)
        out_path = OUT_DIR / filename
        steam_path = TORIBASH_REPLAY_DIR / filename
        out_path.write_text(text, encoding="utf-8")
        shutil.copy2(out_path, steam_path)
        print(f"Wrote {filename}: actions={len(actions)} frames={actions[-1]['frame'] if actions else 0}")

    print("\nCopied to:", TORIBASH_REPLAY_DIR)
    print("Important: V17.1 exports actions-only: no POS/QAT/LINVEL/ANGVEL.")


if __name__ == "__main__":
    main()
