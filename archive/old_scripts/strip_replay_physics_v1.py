#!/usr/bin/env python3
from pathlib import Path
import argparse

PROJECT = Path.home() / "Documents" / "ToribashAI"

DEFAULT_IN_DIR = PROJECT / "models" / "goal_candidates_v1"
DEFAULT_OUT_DIR = PROJECT / "models" / "goal_candidates_v1_stripped"

REMOVE_PREFIXES = (
    "POS ",
    "QAT ",
    "LINVEL ",
    "ANGVEL ",
)


def strip_file(src: Path, dst: Path):
    kept = 0
    removed = 0

    with src.open("r", encoding="utf-8", errors="ignore") as f_in, \
         dst.open("w", encoding="utf-8") as f_out:

        for line in f_in:
            stripped = line.lstrip()

            if stripped.startswith(REMOVE_PREFIXES):
                removed += 1
                continue

            f_out.write(line)
            kept += 1

    return kept, removed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=str, default=str(DEFAULT_IN_DIR))
    parser.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    in_dir = Path(args.in_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()

    if not in_dir.exists():
        raise FileNotFoundError(f"Dossier introuvable: {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    rpls = sorted(in_dir.glob("*.rpl"))

    if not rpls:
        raise FileNotFoundError(f"Aucun .rpl trouvé dans {in_dir}")

    total_removed = 0

    print(f"Input : {in_dir}")
    print(f"Output: {out_dir}")
    print(f"Fichiers: {len(rpls)}")
    print()

    for src in rpls:
        dst = out_dir / src.name
        kept, removed = strip_file(src, dst)
        total_removed += removed
        print(f"{src.name}: kept={kept} removed_physics={removed}")

    print()
    print("Terminé.")
    print(f"Total physics lines removed: {total_removed}")
    print(f"Dossier propre: {out_dir}")


if __name__ == "__main__":
    main()
