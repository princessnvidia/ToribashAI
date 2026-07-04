#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
EVOLUTION = ROOT / "evolution"

SOURCE = EVOLUTION / "champion_xioi_mechanic_v7.json"
OUT_SEED = EVOLUTION / "walk_xioi_seed_v1.json"
OUT_CHAMPION = EVOLUTION / "walk_xioi_champion_v1.json"


def main():
    data = json.loads(SOURCE.read_text(encoding="utf-8"))

    data["name"] = "walk_xioi_seed_v1"
    data["branch"] = "walk_xioi"
    data["source"] = str(SOURCE)
    data["description"] = (
        "Seed walk_xioi basé sur la mécanique Xioi 428 frames, utilisé en boucle."
    )

    OUT_SEED.write_text(json.dumps(data, indent=2), encoding="utf-8")
    OUT_CHAMPION.write_text(json.dumps(data, indent=2), encoding="utf-8")

    print("Seed écrit:", OUT_SEED)
    print("Champion initial écrit:", OUT_CHAMPION)
    print("loop_length:", data.get("loop_length"))
    print("commands:", len(data.get("commands", [])))


if __name__ == "__main__":
    main()
