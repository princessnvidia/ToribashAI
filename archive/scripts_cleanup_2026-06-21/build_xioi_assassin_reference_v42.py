#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from collections import Counter

ROOT = Path.home() / "Documents" / "ToribashAI"
SRC = ROOT / "generated_replays" / "xioi_427_assassincreedhunter_v37.rpl"
OUT = ROOT / "generated_replays" / "xioi_assassin_reference_v42_0_315.json"
SUMMARY = ROOT / "generated_replays" / "xioi_assassin_reference_v42_0_315_summary.json"
FRAME_MAX = 315

POINT_ORDER = [
    (0, "head"),
    (1, "chest"),
    (2, "lumbar"),
    (3, "abs"),
    (4, "left_shoulder"),
    (7, "right_shoulder"),
    (14, "left_hip"),
    (15, "right_hip"),
    (18, "left_foot"),
    (19, "right_foot"),
]

frame_re = re.compile(r"^FRAME\s+(\d+)\s*;")
cmd_re = re.compile(r"^(JOINT|POS|QAT|LINVEL|ANGVEL)\s+(\d+)\s*;\s*(.*)$")


def floats(s: str) -> list[float]:
    out = []
    for tok in s.replace(";", " ").split():
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def ints(s: str) -> list[int]:
    out = []
    for tok in s.replace(";", " ").split():
        try:
            out.append(int(float(tok)))
        except ValueError:
            pass
    return out


def triples(vals: list[float]) -> list[list[float]]:
    return [[vals[i], vals[i+1], vals[i+2]] for i in range(0, len(vals)-2, 3)]


def quads(vals: list[float]) -> list[list[float]]:
    return [[vals[i], vals[i+1], vals[i+2], vals[i+3]] for i in range(0, len(vals)-3, 4)]


def parse_joint_pairs(rest: str) -> list[list[int]]:
    vals = ints(rest)
    pairs = []
    for i in range(0, len(vals)-1, 2):
        j, v = vals[i], vals[i+1]
        if 0 <= j <= 19 and 0 <= v <= 4:
            pairs.append([j, v])
    return pairs


def main() -> None:
    if not SRC.exists():
        raise FileNotFoundError(SRC)

    frames: dict[int, dict] = {}
    current = None
    header = []

    for raw in SRC.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = frame_re.match(raw.strip())
        if m:
            current = int(m.group(1))
            if current <= FRAME_MAX:
                frames.setdefault(current, {"frame": current, "players": {"0": {}, "1": {}}, "raw": []})
            continue

        if current is None:
            header.append(raw)
            continue
        if current > FRAME_MAX:
            continue

        frames[current]["raw"].append(raw)
        c = cmd_re.match(raw.strip())
        if not c:
            continue
        typ, player, rest = c.group(1), c.group(2), c.group(3)
        p = frames[current]["players"].setdefault(player, {})
        if typ == "JOINT":
            pairs = parse_joint_pairs(rest)
            p["joint_pairs"] = pairs
            p["joints"] = {str(j): v for j, v in pairs}
        elif typ in {"POS", "LINVEL", "ANGVEL"}:
            p[typ.lower()] = triples(floats(rest))
        elif typ == "QAT":
            p["qat"] = quads(floats(rest))

    ordered = []
    counts = Counter()
    prev_action = [0] * 20
    for fno in sorted(frames):
        f = frames[fno]
        p0 = f.get("players", {}).get("0", {})
        pairs = p0.get("joint_pairs", [])
        action = [0] * 20
        for j, v in pairs:
            if 0 <= j < 20:
                action[j] = int(v)
                counts[int(v)] += 1
        pos = p0.get("pos", [])
        selected = {}
        for idx, name in POINT_ORDER:
            selected[name] = pos[idx] if idx < len(pos) else [0.0, 0.0, 0.0]
        ordered.append({
            "frame": fno,
            "points": selected,
            "joint_pairs": pairs,
            "action": action,
            "prev_action": prev_action,
        })
        prev_action = action

    data = {
        "version": 42,
        "description": "Xioi assassincreedhunter reference, cropped to the real 0-315 walking segment.",
        "source": str(SRC),
        "frame_min": min(frames) if frames else None,
        "frame_max": max(frames) if frames else None,
        "crop_frame_max": FRAME_MAX,
        "point_order": [name for _, name in POINT_ORDER],
        "frames": ordered,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")

    chest_y0 = ordered[0]["points"].get("chest", [0, 0, 0])[1] if ordered else 0
    chest_y1 = ordered[-1]["points"].get("chest", [0, 0, 0])[1] if ordered else 0
    summary = {
        "version": 42,
        "source": str(SRC),
        "output": str(OUT),
        "frame_count": len(ordered),
        "frame_min": data["frame_min"],
        "frame_max": data["frame_max"],
        "crop_frame_max": FRAME_MAX,
        "action_counts": counts.most_common(),
        "chest_delta_y": chest_y1 - chest_y0,
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Reference:", OUT)
    print("Summary:", SUMMARY)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
