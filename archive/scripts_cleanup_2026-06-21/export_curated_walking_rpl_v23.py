#!/usr/bin/env python3
"""
export_curated_walking_rpl_v23.py

Exporte quelques replays actions-only depuis le dataset marche curaté V23,
pour vérifier visuellement les débuts de marche humains repérés.
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET = ROOT / "datasets" / "ml" / "curated_walking_v23_sequences.jsonl"
OUT_DIR = ROOT / "generated_replays"
TORI_REPLAY = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"

TURNFRAMES = 5
MATCHFRAMES = 900
ENGAGE_Z = 0.0
MAX_ACTIONS_PER_REPLAY = 70


def load_rows():
    rows = []
    with DATASET.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_rpl(name: str, actions: list[dict]) -> Path:
    path = OUT_DIR / f"{name}.rpl"
    lines = []
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
        f"NEWGAME 0;{MATCHFRAMES} {TURNFRAMES} 30 0 0 2 100 0 0 0 0 0 0 0 classic",
        "",
    ]
    for i, act in enumerate(actions[:MAX_ACTIONS_PER_REPLAY]):
        frame = i * TURNFRAMES
        pairs = act.get("pairs", [])
        lines.append(f"FRAME {frame};")
        lines.append(f"# source {act.get('replay','?')} frame {act.get('target_frame','?')}")
        for j, v in pairs:
            lines.append(f"JOINT 0; {int(j)} {int(v)}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main():
    rows = load_rows()
    if not rows:
        raise RuntimeError("Dataset vide. Lance build_curated_walking_dataset_v23.py d'abord.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TORI_REPLAY.mkdir(parents=True, exist_ok=True)

    by_replay = defaultdict(list)
    for r in rows:
        by_replay[r["replay"]].append(r)

    made = []
    for idx, (replay, items) in enumerate(sorted(by_replay.items())[:10], start=1):
        items = sorted(items, key=lambda r: int(r.get("target_frame", 0)))
        actions = [{"pairs": r.get("pairs", []), "replay": replay, "target_frame": r.get("target_frame")} for r in items]
        safe = replay.replace(" ", "_").replace("/", "_").replace(".json", "")[:60]
        name = f"curated_walk_v23_{idx:02d}_{safe}"
        p = write_rpl(name, actions)
        shutil.copy2(p, TORI_REPLAY / p.name)
        made.append(p)
        print("made:", p.name, "actions:", len(actions))

    print("\nCopied to Toribash replay dir:", TORI_REPLAY)


if __name__ == "__main__":
    main()
