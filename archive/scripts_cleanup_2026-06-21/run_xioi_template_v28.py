#!/usr/bin/env python3
"""
run_xioi_template_v28.py

Assistant pratique:
  1. copie le replay Xioi source exact
  2. génère des candidats mutés légers
  3. laisse l'utilisateur les ouvrir dans Toribash > Setup > Replays
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
PY = ROOT / ".venv" / "bin" / "python3"
if not PY.exists():
    PY = Path("python3")

SCRIPTS = ROOT / "scripts"


def main() -> None:
    subprocess.run([str(PY), str(SCRIPTS / "build_xioi_template_v28.py")], check=True)
    subprocess.run([str(PY), str(SCRIPTS / "generate_xioi_template_mutations_v28.py")], check=True)
    print("\nV28 prêt. Ouvre Toribash > Setup > Replays et teste:")
    print("  xioi_source_template_v28.rpl")
    print("  xioi_v28_00_exact_source.rpl")
    print("  xioi_v28_01... à xioi_v28_16...")


if __name__ == "__main__":
    main()
