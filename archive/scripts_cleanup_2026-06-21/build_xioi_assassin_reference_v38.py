#!/usr/bin/env python3
"""
build_xioi_assassin_reference_v38.py

Build a clean reference JSON from the validated Xioi 427-frame replay on
assassincreedhunter.tbm.

Input:
  generated_replays/xioi_427_assassincreedhunter_v37.rpl

Outputs:
  generated_replays/xioi_assassin_reference_v38.json
  generated_replays/xioi_assassin_reference_v38_summary.json

This parser keeps the RPL as source of truth and extracts per-frame JOINT,
POS, QAT, LINVEL and ANGVEL blocks when present.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
IN_RPL = ROOT / "generated_replays" / "xioi_427_assassincreedhunter_v37.rpl"
OUT_JSON = ROOT / "generated_replays" / "xioi_assassin_reference_v38.json"
OUT_SUMMARY = ROOT / "generated_replays" / "xioi_assassin_reference_v38_summary.json"

FRAME_RE = re.compile(r"^FRAME\s+(-?\d+)\s*;")
CMD_RE = re.compile(r"^([A-Z]+)\s+(\d+)\s*;\s*(.*)$")

# Toribash body indexes used by our walking branch. These names are approximate
# but stable enough for reference/proximity scoring.
BODY_POINTS = {
    "head": 0,
    "chest": 1,
    "lumbar": 2,
    "abs": 3,
    "left_shoulder": 5,
    "left_elbow": 6,
    "left_hand": 7,
    "right_shoulder": 8,
    "right_elbow": 9,
    "right_hand": 10,
    "left_hip": 13,
    "right_hip": 14,
    "left_knee": 15,
    "right_knee": 16,
    "left_foot": 19,
    "right_foot": 20,
}


def parse_float_list(s: str) -> list[float]:
    vals: list[float] = []
    for tok in s.replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return vals


def triples(vals: list[float]) -> list[list[float]]:
    return [vals[i : i + 3] for i in range(0, len(vals) - 2, 3)]


def quads(vals: list[float]) -> list[list[float]]:
    return [vals[i : i + 4] for i in range(0, len(vals) - 3, 4)]


def parse_rpl(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    meta: dict[str, Any] = {"source_rpl": str(path)}
    frames: dict[str, dict[str, Any]] = {}
    current: int | None = None

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        if line.startswith("NEWGAME"):
            meta["newgame"] = line
            if "assassincreedhunter" in line.lower():
                meta["mod"] = "assassincreedhunter.tbm"
            continue
        if line.startswith("ENGAGE"):
            meta.setdefault("engage", []).append(line)
            continue
        if line.startswith("FIGHTNAME"):
            meta["fightname"] = line.split(";", 1)[-1].strip() if ";" in line else line
            continue

        fm = FRAME_RE.match(line)
        if fm:
            current = int(fm.group(1))
            frames.setdefault(str(current), {"frame": current, "players": {}})
            continue

        if current is None:
            continue

        cm = CMD_RE.match(line)
        if not cm:
            continue
        cmd, player, rest = cm.group(1), cm.group(2), cm.group(3)
        fr = frames.setdefault(str(current), {"frame": current, "players": {}})
        p = fr["players"].setdefault(player, {})

        if cmd == "JOINT":
            toks = rest.split()
            pairs = []
            for i in range(0, len(toks) - 1, 2):
                try:
                    j = int(float(toks[i])); v = int(float(toks[i + 1]))
                    pairs.append([j, v])
                except ValueError:
                    pass
            if pairs:
                p.setdefault("joint_pairs", []).extend(pairs)
                joints = p.setdefault("joints", {})
                for j, v in pairs:
                    joints[str(j)] = v
        elif cmd == "POS":
            p["pos"] = triples(parse_float_list(rest))
        elif cmd == "QAT":
            p["qat"] = quads(parse_float_list(rest))
        elif cmd == "LINVEL":
            p["linvel"] = triples(parse_float_list(rest))
        elif cmd == "ANGVEL":
            p["angvel"] = triples(parse_float_list(rest))
        else:
            p.setdefault("other", {})[cmd] = rest

    ordered = dict(sorted(frames.items(), key=lambda kv: int(kv[0])))
    return {"name": "xioi_assassin_reference_v38", "version": 38, "metadata": meta, "frames": ordered}


def point(p0: dict[str, Any], idx: int) -> list[float] | None:
    pos = p0.get("pos") or []
    if 0 <= idx < len(pos):
        return pos[idx]
    return None


def main() -> None:
    data = parse_rpl(IN_RPL)
    frames = data["frames"]

    # Add compact walking reference points for fast future scoring/training.
    compact: list[dict[str, Any]] = []
    action_counts = Counter()
    first_y = None
    last_y = None

    for k, fr in frames.items():
        p0 = fr.get("players", {}).get("0", {})
        refs = {name: point(p0, idx) for name, idx in BODY_POINTS.items()}
        joints = p0.get("joint_pairs", [])
        for _, v in joints:
            action_counts[int(v)] += 1
        chest = refs.get("chest") or refs.get("lumbar") or refs.get("abs")
        if chest:
            if first_y is None:
                first_y = chest[1]
            last_y = chest[1]
        compact.append({
            "frame": int(k),
            "points": refs,
            "joint_pairs": joints,
            "joint_count": len(joints),
        })

    data["walking_reference"] = compact
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")

    summary = {
        "version": 38,
        "source": str(IN_RPL),
        "output": str(OUT_JSON),
        "frame_count": len(frames),
        "frame_min": int(next(iter(frames.keys()))) if frames else None,
        "frame_max": int(next(reversed(frames.keys()))) if frames else None,
        "action_counts": action_counts.most_common(),
        "chest_delta_y": None if first_y is None or last_y is None else last_y - first_y,
        "has_pos_frames": sum(1 for fr in frames.values() if fr.get("players", {}).get("0", {}).get("pos")),
        "has_joint_frames": sum(1 for fr in frames.values() if fr.get("players", {}).get("0", {}).get("joint_pairs")),
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Reference:", OUT_JSON)
    print("Summary:", OUT_SUMMARY)
    print(json.dumps(summary, indent=2)[:2000])


if __name__ == "__main__":
    main()
