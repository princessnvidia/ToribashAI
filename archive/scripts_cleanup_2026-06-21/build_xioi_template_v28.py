#!/usr/bin/env python3
"""
build_xioi_template_v28.py

V28 = repartir du replay Xioi lui-même, pas d'un RPL actions-only.

But:
  - trouver le replay Xioi choisi côté JSON parsé
  - retrouver si possible le .rpl brut original dans replays_raw
  - copier un template exact dans generated_replays/ et dans Toribash/replay/

Pourquoi:
  Les actions JOINT seules ne reproduisent pas la marche. La mécanique dépend du
  contexte physique initial: engage, mod, vitesse, position, états initiaux.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
PARKOUR_JSON = ROOT / "datasets" / "parkour_json"
RAW_DIRS = [
    ROOT / "replays_raw" / "parkour_candidate",
    ROOT / "replays_raw",
]
OUT_DIR = ROOT / "generated_replays"
STEAM_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)

# Priorité: le replay Xioi qui a servi au v26/v27 si présent.
PREFERRED_KEYWORDS = [
    "xioi_pk - budokai",
    "xioi",
    "pakourxioi",
]

OUT_TEMPLATE = OUT_DIR / "xioi_source_template_v28.rpl"
OUT_INFO = OUT_DIR / "xioi_source_template_v28_info.json"


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def choose_json() -> Path:
    paths = sorted(PARKOUR_JSON.glob("*.json"))
    if not paths:
        raise FileNotFoundError(PARKOUR_JSON)

    scored: list[tuple[int, Path]] = []
    for p in paths:
        n = norm(p.name)
        score = 0
        for i, kw in enumerate(PREFERRED_KEYWORDS):
            if norm(kw) in n:
                score += 100 - i * 10
        if "budokai" in n:
            score += 50
        if "xioi" in n:
            score += 30
        if score:
            scored.append((score, p))

    if not scored:
        raise RuntimeError("Aucun JSON Xioi trouvé dans datasets/parkour_json")
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]


def find_raw_rpl(json_path: Path) -> Path | None:
    stem = json_path.stem
    # Les JSON commencent souvent par hash_nom. On cherche par hash et fragments du nom.
    hash_part = stem.split("_")[0]
    clean_stem = norm(stem)

    candidates: list[tuple[int, Path]] = []
    for raw_dir in RAW_DIRS:
        if not raw_dir.exists():
            continue
        for p in raw_dir.rglob("*.rpl"):
            n = norm(p.name)
            score = 0
            if hash_part and hash_part.lower() in p.name.lower():
                score += 200
            for token in clean_stem.split():
                if len(token) >= 4 and token in n:
                    score += 10
            if "xioi" in n:
                score += 30
            if "budokai" in n:
                score += 50
            if score:
                candidates.append((score, p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def copy_template(raw_rpl: Path, json_path: Path) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STEAM_REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_rpl, OUT_TEMPLATE)
    shutil.copy2(OUT_TEMPLATE, STEAM_REPLAY_DIR / OUT_TEMPLATE.name)
    return {
        "mode": "raw_rpl_copy",
        "json_source": str(json_path),
        "raw_rpl_source": str(raw_rpl),
        "template": str(OUT_TEMPLATE),
        "steam_copy": str(STEAM_REPLAY_DIR / OUT_TEMPLATE.name),
    }


def main() -> None:
    json_path = choose_json()
    raw = find_raw_rpl(json_path)

    print("Selected JSON:", json_path)
    print("Raw RPL:", raw if raw else "NOT FOUND")

    if not raw:
        raise RuntimeError(
            "Impossible de retrouver le .rpl brut. On peut faire une V28.1 reconstructeur POS/QAT, "
            "mais le plus propre est de récupérer le fichier .rpl source."
        )

    info = copy_template(raw, json_path)
    OUT_INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print("Template copied:", OUT_TEMPLATE)
    print("Steam replay:", STEAM_REPLAY_DIR / OUT_TEMPLATE.name)
    print("Info:", OUT_INFO)


if __name__ == "__main__":
    main()
