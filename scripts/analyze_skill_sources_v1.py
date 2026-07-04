#!/usr/bin/env python3
from pathlib import Path
import json
import re

ROOT = Path.home() / "Documents" / "ToribashAI"
REPLAY_ROOT = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
OUT = ROOT / "datasets" / "skills" / "skill_sources_analysis_v1.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

PATTERNS = [
    "**/*xioi*.rpl",
    "**/*len265*.rpl",
    "**/*v51*.rpl",
    "**/*v53*.rpl",
    "**/*v54*.rpl",
    "**/*v55*.rpl",
]

def analyze(path):
    counts = {
        "FRAME": 0,
        "JOINT": 0,
        "POS": 0,
        "QAT": 0,
        "LINVEL": 0,
        "ANGVEL": 0,
    }
    frames = []
    mod = None
    author = None
    fightname = None

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()

            for k in counts:
                if s.startswith(k + " "):
                    counts[k] += 1

            if s.startswith("FRAME "):
                m = re.search(r"FRAME\s+(-?\d+)", s)
                if m:
                    frames.append(int(m.group(1)))

            if s.startswith("NEWGAME 0;") and mod is None:
                parts = s.split()
                if parts:
                    mod = parts[-1]

            if s.startswith("AUTHOR 0;") and author is None:
                author = s.split(";", 1)[1].strip()

            if s.startswith("FIGHTNAME 0;") and fightname is None:
                fightname = s.split(";", 1)[1].strip()

    return {
        "path": str(path),
        "name": path.name,
        "author": author,
        "fightname": fightname,
        "mod": mod,
        "first_frame": min(frames) if frames else None,
        "last_frame": max(frames) if frames else None,
        "frame_span": (max(frames) - min(frames)) if frames else None,
        "counts": counts,
        "has_physics": counts["POS"] > 0 and counts["QAT"] > 0,
        "has_velocity": counts["LINVEL"] > 0 or counts["ANGVEL"] > 0,
    }

def main():
    found = []
    seen = set()

    for pat in PATTERNS:
        for p in REPLAY_ROOT.glob(pat):
            if p.is_file() and p.suffix.lower() == ".rpl":
                rp = str(p.resolve())
                if rp not in seen:
                    seen.add(rp)
                    found.append(p)

    rows = [analyze(p) for p in sorted(found)]
    rows.sort(key=lambda r: (
        not r["has_physics"],
        -(r["counts"]["FRAME"]),
        r["name"]
    ))

    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"[OK] Sources analysées: {len(rows)}")
    print(f"[OK] Sortie: {OUT}")
    print()
    print("Top 20 sources avec physique:")
    for r in rows[:20]:
        print(
            f"- {r['name']} | frames={r['counts']['FRAME']} "
            f"span={r['first_frame']}→{r['last_frame']} "
            f"joint={r['counts']['JOINT']} pos={r['counts']['POS']} "
            f"vel={r['counts']['LINVEL']}/{r['counts']['ANGVEL']} "
            f"mod={r['mod']}"
        )

if __name__ == "__main__":
    main()
