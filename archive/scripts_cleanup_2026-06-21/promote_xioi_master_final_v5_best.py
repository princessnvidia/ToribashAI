#!/usr/bin/env python3
"""Promote a chosen xioi_master_final_v5_loop_XX.rpl as champion."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
STEAM_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/promote_xioi_master_final_v5_best.py xioi_master_final_v5_loop_XX.rpl")
        raise SystemExit(2)

    name = sys.argv[1]
    src = Path(name)
    if not src.is_absolute():
        src = OUT_DIR / name
    if not src.exists():
        alt = STEAM_REPLAY_DIR / name
        if alt.exists():
            src = alt
        else:
            raise FileNotFoundError(src)

    champion = OUT_DIR / "xioi_master_final_v5_champion.rpl"
    shutil.copy2(src, champion)
    shutil.copy2(champion, STEAM_REPLAY_DIR / champion.name)

    # Also make it the next generic parent for future versions.
    shutil.copy2(champion, OUT_DIR / "xioi_master_latest_champion.rpl")

    print("Promoted:", src)
    print("Champion:", champion)
    print("Copied to Steam replay dir.")


if __name__ == "__main__":
    main()
