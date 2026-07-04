#!/usr/bin/env python3
"""
export_curated_walking_rpl_v23_1.py

Export un replay actions-only par source V23.1 pour inspection visuelle.
Pas de POS/QAT. ENGAGE_Z=0 + classic pour éviter les soucis de mod/replay.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET = ROOT / "datasets" / "ml" / "curated_walking_v23_1_sequences.jsonl"
OUT_DIR = ROOT / "generated_replays"
TORIBASH_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)

MAX_ACTIONS_PER_REPLAY = 90
TURNFRAMES = 5
MATCHFRAMES = 900
ENGAGE_Z = 0.0


def slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    return s[:70] or "source"


def load_rows() -> list[dict[str, Any]]:
    if not DATASET.exists():
        raise FileNotFoundError(DATASET)
    rows = []
    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_rpl(name: str, rows: list[dict[str, Any]], out_path: Path) -> None:
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
    lines.append(f"NEWGAME 0;{MATCHFRAMES} {TURNFRAMES} 30 0 0 2 100 0 0 0 0 0 0 0 classic")
    lines.append("")

    frame = 0
    for idx, row in enumerate(rows[:MAX_ACTIONS_PER_REPLAY]):
        pairs = row.get("pairs") or []
        if not pairs:
            continue
        lines.append(f"FRAME {frame};")
        lines.append(f"# source={row.get('source_file')} target={row.get('target_frame')}")
        for j, v in pairs:
            lines.append(f"JOINT 0; {int(j)} {int(v)}")
        lines.append("")
        frame += TURNFRAMES

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TORIBASH_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_source[r.get("source_file", "unknown")].append(r)

    made = []
    for i, (source, items) in enumerate(sorted(by_source.items()), start=1):
        # garder l'ordre temporel des premières actions
        items.sort(key=lambda r: (int(r.get("target_frame", 0)), int(r.get("start_frame", 0))))
        name = f"curated_walk_v23_1_{i:02d}_{slug(source)}"
        out = OUT_DIR / f"{name}.rpl"
        write_rpl(name, items, out)
        shutil.copy2(out, TORIBASH_REPLAY_DIR / out.name)
        made.append(out.name)
        print("made:", out.name, "actions:", min(len(items), MAX_ACTIONS_PER_REPLAY))

    index = OUT_DIR / "curated_walking_v23_1_exported_replays.json"
    index.write_text(json.dumps({"version": "23.1", "replays": made}, indent=2), encoding="utf-8")
    print("\nCopied to Toribash replay dir:", TORIBASH_REPLAY_DIR)
    print("Count:", len(made))


if __name__ == "__main__":
    main()
