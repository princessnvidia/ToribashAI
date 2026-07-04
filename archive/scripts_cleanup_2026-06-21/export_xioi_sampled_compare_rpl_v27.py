#!/usr/bin/env python3
"""
export_xioi_sampled_compare_rpl_v27.py

Étape 2 / V27:
  Exporte plusieurs .rpl actions-only depuis le dataset Xioi-only.

But:
  - comparer visuellement les actions humaines sampleées
  - vérifier quelles fenêtres sont utiles / nulles hors contexte

Sorties:
  generated_replays/xioi_sampled_compare_*_v27.rpl
  copies dans le dossier replay Steam.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET_PATH = ROOT / "datasets" / "ml" / "xioi_only_v26_sequences.jsonl"
OUT_DIR = ROOT / "generated_replays"
TORIBASH_REPLAY_DIR = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"

TURNFRAMES = 5
MATCHFRAMES = 1200
ENGAGE_Z = 0.0


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", s).strip("_")[:80]


def load_rows() -> list[dict[str, Any]]:
    rows = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("Dataset vide")
    return sorted(rows, key=lambda r: int(r.get("target_frame", 0)))


def action_to_pairs(action: list[int]) -> list[tuple[int, int]]:
    return [(j, int(v)) for j, v in enumerate(action) if int(v) != 0]


def write_rpl(name: str, actions: list[dict[str, Any]]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{safe_name(name)}.rpl"
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
        f"NEWGAME 0;{MATCHFRAMES} {TURNFRAMES} 30 0 0 2 100 0 0 0 0 0 0 0 classic",
        "",
    ]
    for a in actions:
        lines.append(f"FRAME {int(a['frame'])};")
        lines.append(f"# source_frame={a.get('source_frame')} mode={a.get('mode')}")
        for j, v in a.get("pairs", []):
            lines.append(f"JOINT 0; {int(j)} {int(v)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def make_window(rows: list[dict[str, Any]], start_idx: int, steps: int, label: str) -> tuple[str, list[dict[str, Any]]]:
    actions = []
    n = len(rows)
    for step in range(steps):
        r = rows[(start_idx + step) % n]
        pairs = action_to_pairs(r["action"])
        actions.append({
            "frame": step * TURNFRAMES,
            "pairs": pairs,
            "source_frame": r.get("target_frame"),
            "mode": label,
        })
    return label, actions


def main() -> None:
    rows = load_rows()
    print("Rows:", len(rows))
    windows = [
        make_window(rows, 0, 160, "xioi_sampled_compare_01_from_start_v27"),
        make_window(rows, max(0, len(rows)//4), 160, "xioi_sampled_compare_02_quarter_v27"),
        make_window(rows, max(0, len(rows)//2), 160, "xioi_sampled_compare_03_middle_v27"),
        make_window(rows, max(0, len(rows)-180), 180, "xioi_sampled_compare_04_late_v27"),
        make_window(rows, 0, min(260, len(rows)), "xioi_sampled_compare_05_full_loop_v27"),
    ]
    TORIBASH_REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    for name, actions in windows:
        path = write_rpl(name, actions)
        shutil.copy2(path, TORIBASH_REPLAY_DIR / path.name)
        print("made:", path.name, "actions:", len(actions))
    print("Copied to:", TORIBASH_REPLAY_DIR)


if __name__ == "__main__":
    main()
