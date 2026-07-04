#!/usr/bin/env python3
"""
build_xioi_rpl_reference_json_v33.py

Builds a reference JSON from the current walking champion RPL.
This JSON is a "movement map" for V33: joints + physical trajectory.

Input priority:
  generated_replays/xioi_master_final_v5_champion.rpl
  generated_replays/xioi_v30_23_mut.rpl

Output:
  generated_replays/xioi_master_final_v33_reference.json
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean

ROOT = Path.home() / "Documents" / "ToribashAI"
GEN = ROOT / "generated_replays"

CANDIDATES = [
    GEN / "xioi_master_final_v5_champion.rpl",
    GEN / "xioi_v30_23_mut.rpl",
]
OUT = GEN / "xioi_master_final_v33_reference.json"

FRAME_RE = re.compile(r"^FRAME\s+(-?\d+)")
JOINT_RE = re.compile(r"^JOINT\s+(\d+)\s*;\s*(.*)$")
POS_RE = re.compile(r"^POS\s+(\d+)\s*;\s*(.*)$")
QAT_RE = re.compile(r"^QAT\s+(\d+)\s*;\s*(.*)$")
LINVEL_RE = re.compile(r"^LINVEL\s+(\d+)\s*;\s*(.*)$")
ANGVEL_RE = re.compile(r"^ANGVEL\s+(\d+)\s*;\s*(.*)$")

# Toribash body index assumptions from earlier parsed data.
IDX_HEAD = 0
IDX_CHEST = 1
IDX_L_SHOULDER = 5
IDX_R_SHOULDER = 8
IDX_L_HAND = 11
IDX_R_HAND = 12
IDX_L_HIP = 13
IDX_R_HIP = 14
IDX_L_FOOT = 19
IDX_R_FOOT = 20


def floats(s: str) -> list[float]:
    out = []
    for x in s.replace(";", " ").split():
        try:
            out.append(float(x))
        except ValueError:
            pass
    return out


def triples(nums: list[float]) -> list[list[float]]:
    return [nums[i:i+3] for i in range(0, len(nums) - 2, 3)]


def quads(nums: list[float]) -> list[list[float]]:
    return [nums[i:i+4] for i in range(0, len(nums) - 3, 4)]


def parse_joint_pairs(rest: str) -> list[list[int]]:
    vals = []
    for x in rest.replace(";", " ").split():
        try:
            vals.append(int(float(x)))
        except ValueError:
            pass
    pairs = []
    for i in range(0, len(vals) - 1, 2):
        j, v = vals[i], vals[i+1]
        if 0 <= j <= 19 and 0 <= v <= 4:
            pairs.append([j, v])
    return pairs


def pick(points: list[list[float]], idx: int) -> list[float] | None:
    if 0 <= idx < len(points):
        return points[idx]
    return None


def avg_point(points: list[list[float | int | None]]) -> list[float] | None:
    clean = [p for p in points if p is not None]
    if not clean:
        return None
    return [mean([p[0] for p in clean]), mean([p[1] for p in clean]), mean([p[2] for p in clean])]


def parse_rpl(path: Path) -> dict:
    header = []
    frames: dict[int, dict] = {}
    cur: int | None = None
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for line in lines:
        line = line.rstrip("\n")
        m = FRAME_RE.match(line)
        if m:
            cur = int(m.group(1))
            frames.setdefault(cur, {"joints": {}, "raw": []})
            frames[cur]["raw"].append(line)
            continue

        if cur is None:
            header.append(line)
            continue

        f = frames.setdefault(cur, {"joints": {}, "raw": []})
        f["raw"].append(line)

        if (m := JOINT_RE.match(line)):
            player = m.group(1)
            if player == "0":
                for j, v in parse_joint_pairs(m.group(2)):
                    f["joints"][str(j)] = v
        elif (m := POS_RE.match(line)):
            if m.group(1) == "0":
                f["pos"] = triples(floats(m.group(2)))
        elif (m := QAT_RE.match(line)):
            if m.group(1) == "0":
                f["qat"] = quads(floats(m.group(2)))
        elif (m := LINVEL_RE.match(line)):
            if m.group(1) == "0":
                f["linvel"] = triples(floats(m.group(2)))
        elif (m := ANGVEL_RE.match(line)):
            if m.group(1) == "0":
                f["angvel"] = triples(floats(m.group(2)))

    ordered = []
    for fr in sorted(frames):
        f = frames[fr]
        pos = f.get("pos") or []
        head = pick(pos, IDX_HEAD)
        chest = pick(pos, IDX_CHEST)
        l_sh = pick(pos, IDX_L_SHOULDER)
        r_sh = pick(pos, IDX_R_SHOULDER)
        l_hip = pick(pos, IDX_L_HIP)
        r_hip = pick(pos, IDX_R_HIP)
        l_foot = pick(pos, IDX_L_FOOT)
        r_foot = pick(pos, IDX_R_FOOT)
        torso = avg_point([chest, l_sh, r_sh, l_hip, r_hip])
        shoulders = avg_point([l_sh, r_sh])
        hips = avg_point([l_hip, r_hip])

        ordered.append({
            "frame": fr,
            "joints": {str(k): int(v) for k, v in f.get("joints", {}).items()},
            "pos_available": bool(pos),
            "head": head,
            "chest": chest,
            "shoulders": shoulders,
            "hips": hips,
            "torso": torso,
            "left_foot": l_foot,
            "right_foot": r_foot,
        })

    # Guess walking axis from torso displacement.
    with_torso = [x for x in ordered if x.get("torso")]
    if len(with_torso) >= 2:
        p0 = with_torso[0]["torso"]
        p1 = with_torso[-1]["torso"]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        axis = "y" if abs(dy) >= abs(dx) else "x"
        sign = 1 if (dy if axis == "y" else dx) >= 0 else -1
    else:
        axis, sign = "y", 1

    def forward(p):
        if not p:
            return None
        return sign * (p[1] if axis == "y" else p[0])

    f0 = None
    for x in ordered:
        if x.get("torso"):
            f0 = forward(x["torso"])
            break
    for x in ordered:
        x["torso_forward"] = None if f0 is None or not x.get("torso") else forward(x["torso"]) - f0
        x["hips_forward"] = None if f0 is None or not x.get("hips") else forward(x["hips"]) - f0
        x["shoulders_forward"] = None if f0 is None or not x.get("shoulders") else forward(x["shoulders"]) - f0

    return {
        "name": "xioi_master_final_v33_reference",
        "version": 33,
        "source_rpl": str(path),
        "frame_count": len(ordered),
        "forward_axis": axis,
        "forward_sign": sign,
        "header": header,
        "frames": ordered,
    }


def main() -> None:
    src = next((p for p in CANDIDATES if p.exists()), None)
    if not src:
        raise FileNotFoundError("No champion found: " + ", ".join(map(str, CANDIDATES)))
    data = parse_rpl(src)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print("Source:", src)
    print("Output:", OUT)
    print("Frames:", data["frame_count"])
    print("Forward axis:", data["forward_axis"], "sign", data["forward_sign"])


if __name__ == "__main__":
    main()
