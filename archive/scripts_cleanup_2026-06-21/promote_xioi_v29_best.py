#!/usr/bin/env python3
"""
promote_xioi_v29_best.py

Copie un replay V29 choisi visuellement comme nouveau champion.
Usage:
  python3 scripts/promote_xioi_v29_best.py xioi_v29_07_mut.rpl
ou chemin complet.
"""
from __future__ import annotations
import shutil
import sys
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
GEN = ROOT / "generated_replays"
MUT_DIR = GEN / "xioi_v29_mutations"
CHAMPION = GEN / "xioi_v29_champion.rpl"


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/promote_xioi_v29_best.py <xioi_v29_XX_mut.rpl>")
        raise SystemExit(1)
    arg = Path(sys.argv[1])
    if not arg.exists():
        arg = MUT_DIR / sys.argv[1]
    if not arg.exists():
        raise FileNotFoundError(arg)
    shutil.copy2(arg, CHAMPION)
    print("Champion V29:", CHAMPION)

if __name__ == "__main__":
    main()
