#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

ROOT = Path.home() / "Documents/ToribashAI"
GEN = ROOT / "generated_replays"
STEAM = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
REPLAY_ROOT = STEAM / "replay"
REPLAY_PARKOUR = REPLAY_ROOT / "parkour"

OUT_RPL = GEN / "xioi_loop_len265_champion_v51.rpl"
OUT_REF = GEN / "xioi_loop_len265_champion_v51_reference.json"
OUT_SUMMARY = GEN / "xioi_loop_len265_champion_v51_summary.json"

CANDIDATE_PATTERNS = [
    "xioi_loop_phase_v50_1_len265.rpl",
    "xioi_loop_phase_v50_len265.rpl",
    "*len265*.rpl",
]

FIGHTNAME_RE = re.compile(r"^FIGHTNAME\s+0;.*$")
FRAME_RE = re.compile(r"^FRAME\s+(\d+);")
JOINT_RE = re.compile(r"^JOINT\s+0;\s*(.*)$")
NEWGAME_RE = re.compile(r"^NEWGAME\s+0;(.*)$")


def find_source() -> Path:
    roots = [GEN, REPLAY_ROOT, REPLAY_PARKOUR]
    for pat in CANDIDATE_PATTERNS:
        hits = []
        for root in roots:
            if root.exists():
                hits.extend(sorted(root.glob(pat)))
        hits = [p for p in hits if p.is_file() and p.name != OUT_RPL.name]
        if hits:
            # prefer generated_replays, then newest-ish by name/path stable
            hits.sort(key=lambda p: (0 if GEN in p.parents else 1, str(p)))
            return hits[0]
    raise FileNotFoundError(
        "Impossible de trouver le RPL len265. Cherché dans generated_replays, replay/, replay/parkour avec *len265*.rpl"
    )


def fix_fightname(lines: list[str], name: str) -> list[str]:
    out = []
    done = False
    for line in lines:
        if FIGHTNAME_RE.match(line):
            out.append(f"FIGHTNAME 0; {name}")
            done = True
        else:
            out.append(line)
    if not done:
        insert_at = 5 if len(out) > 5 else len(out)
        out.insert(insert_at, f"FIGHTNAME 0; {name}")
    return out


def parse_reference(lines: list[str]) -> dict:
    frames = []
    current = None
    newgame = None
    for line in lines:
        m = NEWGAME_RE.match(line)
        if m:
            newgame = line
        fm = FRAME_RE.match(line)
        if fm:
            current = {"frame": int(fm.group(1)), "pairs": [], "values": [0] * 20}
            frames.append(current)
            continue
        jm = JOINT_RE.match(line)
        if jm and current is not None:
            nums = [int(x) for x in re.findall(r"-?\d+", jm.group(1))]
            pairs = []
            for i in range(0, len(nums) - 1, 2):
                j, v = nums[i], nums[i + 1]
                if 0 <= j < 20 and 0 <= v <= 4:
                    pairs.append([j, v])
                    current["values"][j] = v
            current["pairs"].extend(pairs)
    return {
        "name": "xioi_loop_len265_champion_v51_reference",
        "version": 51,
        "source_rpl": str(OUT_RPL),
        "newgame": newgame,
        "frames": frames,
    }


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    REPLAY_ROOT.mkdir(parents=True, exist_ok=True)
    REPLAY_PARKOUR.mkdir(parents=True, exist_ok=True)

    src = find_source()
    print("Source len265:", src)

    raw_lines = src.read_text(encoding="utf-8", errors="ignore").splitlines()
    lines = fix_fightname(raw_lines, OUT_RPL.stem)
    OUT_RPL.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ref = parse_reference(lines)
    OUT_REF.write_text(json.dumps(ref, indent=2), encoding="utf-8")

    frame_count = len(ref["frames"])
    action_counts = {}
    active_dist = {}
    for fr in ref["frames"]:
        active = len(fr["pairs"])
        active_dist[active] = active_dist.get(active, 0) + 1
        for _, v in fr["pairs"]:
            action_counts[v] = action_counts.get(v, 0) + 1
    summary = {
        "version": 51,
        "source": str(src),
        "champion": str(OUT_RPL),
        "reference": str(OUT_REF),
        "frame_count": frame_count,
        "frame_min": min((f["frame"] for f in ref["frames"]), default=None),
        "frame_max": max((f["frame"] for f in ref["frames"]), default=None),
        "action_counts": sorted(action_counts.items()),
        "active_distribution": sorted(active_dist.items()),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    for dst_dir in (REPLAY_ROOT, REPLAY_PARKOUR):
        dst = dst_dir / OUT_RPL.name
        shutil.copy2(OUT_RPL, dst)
        print("Copied to:", dst)

    print("Champion:", OUT_RPL)
    print("Reference:", OUT_REF)
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
