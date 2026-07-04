#!/usr/bin/env python3
from pathlib import Path

REPLAY_DIR = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"

for path in sorted(REPLAY_DIR.glob("xioi_v33_g*.rpl")):
    name = path.stem
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    done = False
    out = []
    for line in lines:
        if line.startswith("FIGHTNAME 0;"):
            out.append(f"FIGHTNAME 0; {name}")
            done = True
        else:
            out.append(line)

    if not done:
        insert_at = 5 if len(out) > 5 else len(out)
        out.insert(insert_at, f"FIGHTNAME 0; {name}")

    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    path.touch()
    print("fixed", path.name)
