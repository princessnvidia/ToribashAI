#!/usr/bin/env python3
from pathlib import Path
import json

PARSED = Path.home() / "Documents/ToribashAI" / "parsed"

count = 0

for path in sorted(PARSED.glob("*.json")):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata", {})
    mod = metadata.get("mod")

    if not mod or mod == "UNKNOWN":
        print(path.stem)
        print("  fightname:", metadata.get("fightname"))
        print("  newgame_raw:", metadata.get("newgame_raw"))
        print()

        count += 1

        if count >= 20:
            break

print("Exemples trouvés:", count)
