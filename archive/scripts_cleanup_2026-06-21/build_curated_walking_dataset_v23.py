#!/usr/bin/env python3
"""
build_curated_walking_dataset_v23.py

V23 = dataset marche curaté.

On arrête d'essayer d'extraire la marche depuis tous les parkour.
On utilise uniquement les replays que Vio a repérés visuellement comme ayant un début de marche :
  - xioi - chichaehaa
  - xioi - ff
  - swexx - divine
  - flarkour
  - karbn - raid
  - Mack - P - Clay pigeon
  - the kurr of treasure hunting

Sorties :
  datasets/ml/curated_walking_v23_sequences.jsonl
  generated_replays/curated_walking_v23_summary.json
  generated_replays/curated_walking_v23_sources.json
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
PARKOUR_JSON = ROOT / "datasets" / "parkour_json"
OUT_DATASET = ROOT / "datasets" / "ml" / "curated_walking_v23_sequences.jsonl"
OUT_SUMMARY = ROOT / "generated_replays" / "curated_walking_v23_summary.json"
OUT_SOURCES = ROOT / "generated_replays" / "curated_walking_v23_sources.json"

CURATED_NAMES = [
    "xioi - chichaehaa",
    "xioi - ff",
    "swexx - divine",
    "flarkour",
    "karbn - raid",
    "mack - p - clay pigeon",
    "the kurr of treasure hunting",
]

# On extrait le début uniquement. Les débuts visibles de marche sont généralement avant le trick.
MAX_FRAME = 220
SEQ_LEN = 8
MIN_ACTION_FRAMES = 4

# Format modèle déjà utilisé ailleurs : 20 joints.
JOINT_COUNT = 20

# Indices de body parts dans POS Toribash, approximatifs mais cohérents avec nos usages précédents.
# On garde surtout des features relatives, pas besoin d'une sémantique parfaite.
HEAD_IDX = 0
CHEST_IDX = 2
L_SHOULDER_IDX = 4
R_SHOULDER_IDX = 7
L_FOOT_IDX = 18
R_FOOT_IDX = 19
HIP_IDXS = [14, 15]


def norm(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").replace("-", " - ").split())


def find_curated_files() -> list[Path]:
    files = sorted(PARKOUR_JSON.glob("*.json"))
    found: list[Path] = []
    seen = set()
    for wanted in CURATED_NAMES:
        wanted_n = norm(wanted)
        matches = []
        for p in files:
            name = norm(p.stem)
            if wanted_n in name:
                matches.append(p)
        # fallback: all words must appear
        if not matches:
            words = [w for w in wanted_n.replace("-", " ").split() if len(w) > 1]
            for p in files:
                name = norm(p.stem)
                if all(w in name for w in words):
                    matches.append(p)
        for p in matches:
            if p not in seen:
                found.append(p)
                seen.add(p)
    return found


def sorted_frames(data: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    frames = data.get("frames", {})
    if isinstance(frames, dict):
        out = []
        for k, v in frames.items():
            try:
                out.append((int(k), v))
            except Exception:
                continue
        return sorted(out, key=lambda x: x[0])
    if isinstance(frames, list):
        return [(i, f) for i, f in enumerate(frames) if isinstance(f, dict)]
    return []


def p0(frame: dict[str, Any]) -> dict[str, Any]:
    return frame.get("players", {}).get("0", {}) or frame.get("players", {}).get(0, {}) or {}


def get_pos(player: dict[str, Any], idx: int) -> list[float] | None:
    pos = player.get("pos")
    if not isinstance(pos, list) or idx >= len(pos):
        return None
    v = pos[idx]
    if not isinstance(v, list) or len(v) < 3:
        return None
    try:
        return [float(v[0]), float(v[1]), float(v[2])]
    except Exception:
        return None


def vsub(a: list[float], b: list[float]) -> list[float]:
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def dist2_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def joints_vec(player: dict[str, Any]) -> list[int]:
    vec = [0] * JOINT_COUNT
    joints = player.get("joints", {})
    if isinstance(joints, dict):
        for k, v in joints.items():
            try:
                j = int(k)
                if 0 <= j < JOINT_COUNT:
                    vec[j] = int(v)
            except Exception:
                pass
    pairs = player.get("joint_pairs", [])
    if isinstance(pairs, list):
        for pair in pairs:
            if isinstance(pair, list) and len(pair) >= 2:
                try:
                    j = int(pair[0])
                    v = int(pair[1])
                    if 0 <= j < JOINT_COUNT:
                        vec[j] = v
                except Exception:
                    pass
    return vec


def pairs_from_vec(vec: list[int]) -> list[list[int]]:
    return [[i, int(v)] for i, v in enumerate(vec) if int(v) != 0]


def feature_vec(frame_no: int, player: dict[str, Any], origin: list[float], prev_player: dict[str, Any] | None) -> list[float] | None:
    head = get_pos(player, HEAD_IDX)
    chest = get_pos(player, CHEST_IDX)
    ls = get_pos(player, L_SHOULDER_IDX)
    rs = get_pos(player, R_SHOULDER_IDX)
    lf = get_pos(player, L_FOOT_IDX)
    rf = get_pos(player, R_FOOT_IDX)
    hips = [get_pos(player, i) for i in HIP_IDXS]
    hips = [h for h in hips if h]
    if not all([head, chest, ls, rs, lf, rf]) or not hips:
        return None

    hip = [sum(h[i] for h in hips) / len(hips) for i in range(3)]
    shoulder_mid = [(ls[i] + rs[i]) / 2.0 for i in range(3)]
    shoulder_width = dist2_xy(ls, rs)
    foot_gap = dist2_xy(lf, rf)
    foot_z_min = min(lf[2], rf[2])
    foot_z_diff = abs(lf[2] - rf[2])

    prev_speed = [0.0, 0.0, 0.0]
    if prev_player is not None:
        prev_hip = None
        prev_hips = [get_pos(prev_player, i) for i in HIP_IDXS]
        prev_hips = [h for h in prev_hips if h]
        if prev_hips:
            prev_hip = [sum(h[i] for h in prev_hips) / len(prev_hips) for i in range(3)]
        if prev_hip:
            prev_speed = vsub(hip, prev_hip)

    jv = joints_vec(player)
    active_count = sum(1 for v in jv if v != 0)

    # Compact state orienté marche : positions relatives + hauteur/stabilité + joints actuels normalisés.
    feats = []
    feats.extend([(hip[0] - origin[0]) / 30.0, (hip[1] - origin[1]) / 30.0, hip[2] / 30.0])
    feats.extend([(head[0] - hip[0]) / 10.0, (head[1] - hip[1]) / 10.0, (head[2] - hip[2]) / 10.0])
    feats.extend([(shoulder_mid[0] - hip[0]) / 10.0, (shoulder_mid[1] - hip[1]) / 10.0, shoulder_mid[2] / 30.0])
    feats.extend([(lf[0] - hip[0]) / 10.0, (lf[1] - hip[1]) / 10.0, (lf[2] - foot_z_min) / 5.0])
    feats.extend([(rf[0] - hip[0]) / 10.0, (rf[1] - hip[1]) / 10.0, (rf[2] - foot_z_min) / 5.0])
    feats.extend([shoulder_width / 10.0, foot_gap / 10.0, foot_z_diff / 5.0])
    feats.extend([prev_speed[0] / 5.0, prev_speed[1] / 5.0, prev_speed[2] / 5.0])
    feats.append(active_count / 20.0)
    feats.extend([v / 4.0 for v in jv])
    return feats


def extract_from_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = [(n, f) for n, f in sorted_frames(data) if n <= MAX_FRAME]
    usable = []
    prev = None
    origin = None

    for n, f in frames:
        player = p0(f)
        hip_candidates = [get_pos(player, i) for i in HIP_IDXS]
        hip_candidates = [h for h in hip_candidates if h]
        if not hip_candidates:
            continue
        hip = [sum(h[i] for h in hip_candidates) / len(hip_candidates) for i in range(3)]
        if origin is None:
            origin = hip
        feats = feature_vec(n, player, origin, prev)
        jv = joints_vec(player)
        if feats is None:
            prev = player
            continue
        usable.append({
            "frame": n,
            "state": feats,
            "action": jv,
            "pairs": pairs_from_vec(jv),
            "active_count": sum(1 for v in jv if v != 0),
        })
        prev = player

    rows = []
    for i in range(0, max(0, len(usable) - SEQ_LEN)):
        seq = usable[i:i + SEQ_LEN]
        target = usable[i + SEQ_LEN]
        if sum(1 for x in seq if x["active_count"] > 0) < MIN_ACTION_FRAMES:
            continue
        rows.append({
            "source_file": str(path),
            "replay": path.name,
            "start_frame": seq[0]["frame"],
            "target_frame": target["frame"],
            "state_seq": [x["state"] for x in seq],
            "action": target["action"],
            "pairs": target["pairs"],
        })

    info = {
        "file": str(path),
        "usable_frames": len(usable),
        "sequences": len(rows),
        "first_frame": usable[0]["frame"] if usable else None,
        "last_frame": usable[-1]["frame"] if usable else None,
    }
    return rows, info


def main() -> None:
    files = find_curated_files()
    print("Curated wanted:")
    for n in CURATED_NAMES:
        print("  -", n)
    print("\nFound files:", len(files))
    for p in files:
        print(" ", p.name)

    if not files:
        raise RuntimeError("Aucun replay curaté trouvé. Vérifie les noms dans datasets/parkour_json.")

    OUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    all_rows = []
    sources = []
    action_counts = Counter()
    for p in files:
        rows, info = extract_from_file(p)
        all_rows.extend(rows)
        sources.append(info)
        for r in rows:
            for v in r["action"]:
                action_counts[int(v)] += 1
        print(f"{p.name}: usable={info['usable_frames']} sequences={info['sequences']} frames={info['first_frame']}-{info['last_frame']}")

    with OUT_DATASET.open("w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "version": 23,
        "name": "curated_walking_v23",
        "seq_len": SEQ_LEN,
        "max_frame": MAX_FRAME,
        "found_files": len(files),
        "sequences": len(all_rows),
        "state_dim": len(all_rows[0]["state_seq"][0]) if all_rows else None,
        "action_dim": JOINT_COUNT,
        "action_value_counts": dict(action_counts),
        "sources": sources,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_SOURCES.write_text(json.dumps({"wanted": CURATED_NAMES, "files": [str(p) for p in files], "sources": sources}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSaved dataset:", OUT_DATASET)
    print("Saved summary:", OUT_SUMMARY)
    print("Sequences:", len(all_rows))
    print("State dim:", summary["state_dim"])
    print("Action counts:", action_counts.most_common())


if __name__ == "__main__":
    main()
