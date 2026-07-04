#!/usr/bin/env python3
"""
promote_xioi_v30_best.py

Promote une mutation V30 choisie visuellement en champion officiel.
Usage:
  python3 scripts/promote_xioi_v30_best.py xioi_v30_23_mut.rpl
ou avec un chemin complet.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
TORIBASH_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)

CHAMPION = OUT_DIR / "xioi_v30_champion.rpl"
NEXT_SEED = OUT_DIR / "xioi_v31_seed_from_v30.rpl"


def resolve(name: str) -> Path:
    p = Path(name).expanduser()
    if p.exists():
        return p
    for base in (OUT_DIR, TORIBASH_REPLAY_DIR):
        q = base / name
        if q.exists():
            return q
    raise FileNotFoundError(name)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/promote_xioi_v30_best.py xioi_v30_XX_mut.rpl")
        raise SystemExit(2)

    src = resolve(sys.argv[1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TORIBASH_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(src, CHAMPION)
    shutil.copy2(src, NEXT_SEED)
    shutil.copy2(src, TORIBASH_REPLAY_DIR / CHAMPION.name)
    shutil.copy2(src, TORIBASH_REPLAY_DIR / NEXT_SEED.name)

    print("Promoted:", src)
    print("Champion:", CHAMPION)
    print("Next seed:", NEXT_SEED)
    print("Copied to Toribash replay dir.")


if __name__ == "__main__":
    main()
