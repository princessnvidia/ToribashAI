#!/usr/bin/env python3
"""
convert_xioi_427_to_assassincreedhunter_v37.py

Convertit le replay Xioi source/template en gardant toute la mécanique complète
(POS/QAT/LINVEL/ANGVEL/JOINT), mais force le NEWGAME vers assassincreedhunter.tbm.

Sorties :
  generated_replays/xioi_427_assassincreedhunter_v37.rpl
  replay/parkour/xioi_427_assassincreedhunter_v37.rpl
  replay/xioi_427_assassincreedhunter_v37.rpl
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
GEN = ROOT / "generated_replays"
STEAM_TB = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
REPLAY_ROOT = STEAM_TB / "replay"
REPLAY_PARKOUR = REPLAY_ROOT / "parkour"

OUT_NAME = "xioi_427_assassincreedhunter_v37.rpl"
OUT_PATH = GEN / OUT_NAME
TARGET_MOD = "assassincreedhunter.tbm"
TARGET_FIGHTNAME = "xioi_427_assassincreedhunter_v37"

CANDIDATES = [
    GEN / "xioi_source_template_v28.rpl",
    GEN / "xioi_v30_23_mut.rpl",
    GEN / "xioi_master_final_v5_champion.rpl",
    GEN / "xioi_same_foot_loop_walk_v35_1_base.rpl",
]


def find_source() -> Path:
    for p in CANDIDATES:
        if p.exists():
            return p
    # fallback: any likely xioi source
    matches = sorted(GEN.glob("*xioi*source*template*.rpl")) + sorted(GEN.glob("*xioi*427*.rpl"))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        "Impossible de trouver un replay source Xioi. Cherché:\n" +
        "\n".join(str(p) for p in CANDIDATES)
    )


def patch_newgame(line: str) -> str:
    # Remplace seulement le mod final si NEWGAME existe déjà.
    # Les NEWGAME Toribash ont souvent le mod en dernier token.
    if not line.startswith("NEWGAME"):
        return line
    parts = line.rstrip("\n").split()
    if len(parts) >= 2:
        # Si le dernier token ressemble à un mod ou classic, remplace-le.
        if parts[-1].endswith(".tbm") or parts[-1] == "classic" or "/" in parts[-1]:
            parts[-1] = TARGET_MOD
            return " ".join(parts) + "\n"
    # Fallback robuste, garde les paramètres usuels.
    return f"NEWGAME 0;2000 5 30 0 0 2 100 0 0 0 0 0 0 0 {TARGET_MOD}\n"


def convert() -> Path:
    src = find_source()
    print("Source:", src)

    text = src.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines(keepends=True)

    out: list[str] = []
    saw_fight = False
    saw_newgame = False

    for line in lines:
        if line.startswith("FIGHTNAME 0;"):
            out.append(f"FIGHTNAME 0; {TARGET_FIGHTNAME}\n")
            saw_fight = True
        elif line.startswith("NEWGAME"):
            out.append(patch_newgame(line))
            saw_newgame = True
        else:
            out.append(line)

    # Ajoute FIGHTNAME si absent, après VERSION si possible.
    if not saw_fight:
        insert = 0
        for i, line in enumerate(out):
            if line.startswith("VERSION"):
                insert = i + 1
                break
        out.insert(insert, f"FIGHTNAME 0; {TARGET_FIGHTNAME}\n")

    # Ajoute NEWGAME si absent avant le premier FRAME.
    if not saw_newgame:
        insert = 0
        for i, line in enumerate(out):
            if line.startswith("FRAME"):
                insert = i
                break
        out.insert(insert, f"NEWGAME 0;2000 5 30 0 0 2 100 0 0 0 0 0 0 0 {TARGET_MOD}\n\n")

    # Limite optionnelle aux 427 premières frames si le replay source est plus long.
    # On garde tout jusqu'à FRAME 427 inclus puis on stop aux frames suivantes.
    trimmed: list[str] = []
    current_frame = None
    for line in out:
        m = re.match(r"FRAME\s+(\d+);", line)
        if m:
            current_frame = int(m.group(1))
            if current_frame > 427:
                break
        trimmed.append(line)

    GEN.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("".join(trimmed), encoding="utf-8")

    for dst_dir in (REPLAY_ROOT, REPLAY_PARKOUR):
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT_PATH, dst_dir / OUT_NAME)
        print("Copied:", dst_dir / OUT_NAME)

    print("Wrote:", OUT_PATH)
    print("Mod:", TARGET_MOD)
    print("Fightname:", TARGET_FIGHTNAME)
    return OUT_PATH


if __name__ == "__main__":
    convert()
