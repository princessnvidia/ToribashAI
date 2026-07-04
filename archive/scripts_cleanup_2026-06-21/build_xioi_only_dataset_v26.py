#!/usr/bin/env python3
"""
build_xioi_only_dataset_v26.py

Branche V26 = Xioi-only walking GRU.

Objectif:
  - trouver le meilleur replay Xioi dans datasets/parkour_json
  - extraire uniquement ses premières ~427 frames / actions
  - construire un dataset séquence propre pour entraîner un GRU marche Xioi

Sorties:
  datasets/ml/xioi_only_v26_sequences.jsonl
  generated_replays/xioi_only_v26_summary.json
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
PARKOUR_JSON = ROOT / "datasets" / "parkour_json"
OUT_DATASET = ROOT / "datasets" / "ml" / "xioi_only_v26_sequences.jsonl"
OUT_SUMMARY = ROOT / "generated_replays" / "xioi_only_v26_summary.json"

SEQ_LEN = 8
MAX_FRAME = 427
ACTION_DIM = 20

# On préfère les vrais fichiers Xioi trouvés dans ton dataset.
PREFERRED_KEYWORDS = [
    "pakourxioi",
    "xioi_pk",
    "xioi-pk",
    "xioi pk",
    "xioi",
]

# Indices de corps utiles pour l'état. On garde un état compact: torse/tête/épaules/hanches/pieds.
# Les indices suivent le tableau POS Toribash parsé; on reste robuste si certains indices manquent.
BODY_IDXS = [0, 1, 2, 3, 4, 5, 8, 9, 14, 15, 16, 17, 18, 19, 20]


def norm_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sorted_frames(data: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    frames = data.get("frames", {})
    out = []
    if isinstance(frames, dict):
        for k, v in frames.items():
            try:
                out.append((int(k), v))
            except Exception:
                pass
    elif isinstance(frames, list):
        for i, v in enumerate(frames):
            if isinstance(v, dict):
                out.append((int(v.get("frame", i)), v))
    out.sort(key=lambda x: x[0])
    return out


def player0(frame: dict[str, Any]) -> dict[str, Any]:
    return frame.get("players", {}).get("0", {}) or frame.get("players", {}).get(0, {}) or {}


def frame_has_action(frame: dict[str, Any]) -> bool:
    p0 = player0(frame)
    return bool(p0.get("joint_pairs") or p0.get("joints"))


def action_from_frame(frame: dict[str, Any]) -> list[int]:
    action = [0] * ACTION_DIM
    p0 = player0(frame)
    pairs = p0.get("joint_pairs")
    if isinstance(pairs, list):
        for pair in pairs:
            if isinstance(pair, list) and len(pair) >= 2:
                try:
                    j = int(pair[0])
                    v = int(pair[1])
                except Exception:
                    continue
                if 0 <= j < ACTION_DIM and 0 <= v <= 4:
                    action[j] = v
    elif isinstance(p0.get("joints"), dict):
        for k, v in p0["joints"].items():
            try:
                j = int(k)
                val = int(v)
            except Exception:
                continue
            if 0 <= j < ACTION_DIM and 0 <= val <= 4:
                action[j] = val
    return action


def state_from_frame(frame: dict[str, Any], origin: list[float] | None = None) -> list[float] | None:
    p0 = player0(frame)
    pos = p0.get("pos")
    if not isinstance(pos, list) or not pos:
        return None

    # Origine = premier body part si dispo, pour réduire la dépendance au mod/hauteur.
    if origin is None:
        try:
            origin = [float(pos[0][0]), float(pos[0][1]), float(pos[0][2])]
        except Exception:
            origin = [0.0, 0.0, 0.0]

    feats: list[float] = []
    for idx in BODY_IDXS:
        if idx < len(pos) and isinstance(pos[idx], list) and len(pos[idx]) >= 3:
            try:
                feats.extend([
                    float(pos[idx][0]) - origin[0],
                    float(pos[idx][1]) - origin[1],
                    float(pos[idx][2]) - origin[2],
                ])
            except Exception:
                feats.extend([0.0, 0.0, 0.0])
        else:
            feats.extend([0.0, 0.0, 0.0])
    return feats


def score_xioi_candidate(path: Path, data: dict[str, Any]) -> tuple[int, int, int]:
    name = norm_name(path.name)
    keyword_score = 0
    for i, kw in enumerate(PREFERRED_KEYWORDS):
        if norm_name(kw) in name:
            keyword_score += 100 - i * 5
    frs = sorted_frames(data)
    early = [(n, f) for n, f in frs if n <= MAX_FRAME]
    usable = sum(1 for _, f in early if state_from_frame(f) is not None)
    actions = sum(1 for _, f in early if frame_has_action(f))
    return keyword_score, usable, actions


def choose_xioi_file() -> Path:
    candidates = []
    for path in PARKOUR_JSON.glob("*.json"):
        name = norm_name(path.name)
        if "xioi" not in name:
            continue
        try:
            data = load_json(path)
            score = score_xioi_candidate(path, data)
            candidates.append((score, path))
        except Exception:
            continue
    if not candidates:
        raise FileNotFoundError("Aucun fichier Xioi trouvé dans datasets/parkour_json")
    candidates.sort(key=lambda x: x[0], reverse=True)
    print("Xioi candidates:")
    for score, path in candidates[:12]:
        print(" ", score, path.name)
    return candidates[0][1]


def compact_rows(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    rows: list[dict[str, Any]] = []
    origin: list[float] | None = None

    for frame_no, frame in sorted_frames(data):
        if frame_no > MAX_FRAME:
            continue
        p0 = player0(frame)
        pos = p0.get("pos")
        if origin is None and isinstance(pos, list) and pos:
            try:
                origin = [float(pos[0][0]), float(pos[0][1]), float(pos[0][2])]
            except Exception:
                origin = [0.0, 0.0, 0.0]

        state = state_from_frame(frame, origin)
        if state is None:
            continue
        action = action_from_frame(frame)
        rows.append({
            "frame": frame_no,
            "state": state,
            "action": action,
            "source": path.name,
        })

    return rows


def main() -> None:
    xioi_path = choose_xioi_file()
    rows = compact_rows(xioi_path)
    if len(rows) <= SEQ_LEN:
        raise RuntimeError(f"Pas assez de frames utilisables dans {xioi_path.name}: {len(rows)}")

    sequences = []
    for i in range(len(rows) - SEQ_LEN):
        seq = rows[i:i + SEQ_LEN]
        target = rows[i + SEQ_LEN]
        sequences.append({
            "source": xioi_path.name,
            "seq_start_frame": seq[0]["frame"],
            "target_frame": target["frame"],
            "state_seq": [r["state"] for r in seq],
            "action": target["action"],
        })

    OUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DATASET.open("w", encoding="utf-8") as f:
        for s in sequences:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    action_counts = Counter()
    active_counts = Counter()
    for s in sequences:
        active = 0
        for v in s["action"]:
            action_counts[int(v)] += 1
            if int(v) != 0:
                active += 1
        active_counts[active] += 1

    summary = {
        "version": 26,
        "source": str(xioi_path),
        "source_name": xioi_path.name,
        "rows": len(rows),
        "sequences": len(sequences),
        "seq_len": SEQ_LEN,
        "state_dim": len(rows[0]["state"]),
        "action_dim": ACTION_DIM,
        "max_frame": MAX_FRAME,
        "frame_min": rows[0]["frame"],
        "frame_max": rows[-1]["frame"],
        "action_counts": action_counts.most_common(),
        "active_counts": active_counts.most_common(),
        "dataset": str(OUT_DATASET),
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nChosen Xioi:", xioi_path.name)
    print("Rows:", len(rows))
    print("Sequences:", len(sequences))
    print("State dim:", summary["state_dim"])
    print("Action counts:", action_counts.most_common())
    print("Active counts:", active_counts.most_common(12))
    print("Saved dataset:", OUT_DATASET)
    print("Saved summary:", OUT_SUMMARY)


if __name__ == "__main__":
    main()
