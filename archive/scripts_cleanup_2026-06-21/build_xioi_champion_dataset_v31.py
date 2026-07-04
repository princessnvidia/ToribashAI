#!/usr/bin/env python3
"""
build_xioi_champion_dataset_v31.py

V31: construit un dataset d'imitation depuis le champion Xioi V30 promu.

Entrée préférée:
  generated_replays/xioi_v30_champion.rpl
Fallbacks:
  generated_replays/xioi_v30_23_mut.rpl
  generated_replays/xioi_v29_champion.rpl

Sortie:
  datasets/ml/xioi_champion_v31_sequences.jsonl
  generated_replays/xioi_champion_v31_summary.json

Note:
  On lit les JOINT du .rpl. Les POS/QAT restent utiles pour le replay source,
  mais le GRU apprend ici la chorégraphie d'actions du champion.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DATASET = ROOT / "datasets" / "ml" / "xioi_champion_v31_sequences.jsonl"
OUT_SUMMARY = ROOT / "generated_replays" / "xioi_champion_v31_summary.json"

SEQ_LEN = 8
ACTION_DIM = 20
CLASSES = 5
TURNFRAMES = 5
MAX_FRAME = 520

CANDIDATES = [
    ROOT / "generated_replays" / "xioi_v30_champion.rpl",
    ROOT / "generated_replays" / "xioi_v30_23_mut.rpl",
    ROOT / "generated_replays" / "xioi_v29_champion.rpl",
    ROOT / "generated_replays" / "xioi_source_template_v28.rpl",
]

FRAME_RE = re.compile(r"^FRAME\s+(-?\d+)\s*;")
JOINT_RE = re.compile(r"^JOINT\s+0\s*;\s*(\d+)\s+([0-4])")
NEWGAME_RE = re.compile(r"^NEWGAME\s+0\s*;(.*)$")
ENGAGE_RE = re.compile(r"^ENGAGE\s+(\d+)\s*;(.*)$")


def find_champion() -> Path:
    for p in CANDIDATES:
        if p.exists():
            return p
    hits = sorted((ROOT / "generated_replays").glob("*v30*champion*.rpl"))
    if hits:
        return hits[-1]
    hits = sorted((ROOT / "generated_replays").glob("*v30*23*mut*.rpl"))
    if hits:
        return hits[-1]
    raise FileNotFoundError("Aucun champion V30 trouvé dans generated_replays")


def parse_rpl(path: Path) -> dict[str, Any]:
    frames: dict[int, list[list[int]]] = {}
    current_frame: int | None = None
    header: list[str] = []
    newgame = None
    engage: dict[str, str] = {}

    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        m = FRAME_RE.match(line)
        if m:
            current_frame = int(m.group(1))
            frames.setdefault(current_frame, [])
            continue
        if current_frame is None:
            header.append(raw)
            mng = NEWGAME_RE.match(line)
            if mng:
                newgame = mng.group(1).strip()
            meng = ENGAGE_RE.match(line)
            if meng:
                engage[meng.group(1)] = meng.group(2).strip()
            continue
        jm = JOINT_RE.match(line)
        if jm and current_frame is not None:
            j = int(jm.group(1))
            v = int(jm.group(2))
            if 0 <= j < ACTION_DIM:
                frames.setdefault(current_frame, []).append([j, v])

    if not frames:
        raise RuntimeError(f"Aucune frame JOINT trouvée dans {path}")

    return {
        "path": str(path),
        "frames": frames,
        "header": header,
        "newgame": newgame,
        "engage": engage,
    }


def sparse_action(pairs: list[list[int]]) -> list[int]:
    a = [0] * ACTION_DIM
    for j, v in pairs:
        if 0 <= int(j) < ACTION_DIM:
            a[int(j)] = max(0, min(4, int(v)))
    return a


def state_from_action(action: list[int], frame: int) -> list[float]:
    # 20 joints * 5 one-hot classes + phase/sin/cos + normalized frame = 104 dims.
    out: list[float] = []
    for v in action:
        for c in range(CLASSES):
            out.append(1.0 if int(v) == c else 0.0)
    phase = (frame % 80) / 80.0
    out.append(float(frame) / max(1.0, float(MAX_FRAME)))
    out.append(math.sin(2.0 * math.pi * phase))
    out.append(math.cos(2.0 * math.pi * phase))
    out.append(phase)
    return out


def main() -> None:
    champion = find_champion()
    parsed = parse_rpl(champion)
    frames_raw: dict[int, list[list[int]]] = parsed["frames"]
    frame_keys = sorted(f for f in frames_raw if 0 <= f <= MAX_FRAME)
    if len(frame_keys) <= SEQ_LEN:
        raise RuntimeError("Pas assez de frames pour construire des séquences")

    actions_by_frame = {f: sparse_action(frames_raw.get(f, [])) for f in frame_keys}
    states_by_frame = {f: state_from_action(actions_by_frame[f], f) for f in frame_keys}

    rows = []
    for i in range(SEQ_LEN, len(frame_keys)):
        seq_frames = frame_keys[i - SEQ_LEN:i]
        target_frame = frame_keys[i]
        rows.append({
            "version": 31,
            "source": champion.name,
            "source_path": str(champion),
            "seq_frames": seq_frames,
            "target_frame": target_frame,
            "state_seq": [states_by_frame[f] for f in seq_frames],
            "action": actions_by_frame[target_frame],
            "pairs": frames_raw.get(target_frame, []),
        })

    OUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    with OUT_DATASET.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    counts = Counter()
    pair_counts = Counter()
    active_counts = Counter()
    for r in rows:
        active_counts[sum(1 for v in r["action"] if v != 0)] += 1
        for j, v in enumerate(r["action"]):
            counts[int(v)] += 1
            if v != 0:
                pair_counts[(j, int(v))] += 1

    summary = {
        "version": 31,
        "champion": str(champion),
        "dataset": str(OUT_DATASET),
        "sequences": len(rows),
        "seq_len": SEQ_LEN,
        "state_dim": len(rows[0]["state_seq"][0]) if rows else 0,
        "frame_min": min(frame_keys),
        "frame_max": max(frame_keys),
        "frames_used": len(frame_keys),
        "action_counts": counts.most_common(),
        "active_counts": active_counts.most_common(),
        "top_pairs": [([j, v], c) for (j, v), c in pair_counts.most_common(30)],
        "source_newgame": parsed.get("newgame"),
        "source_engage": parsed.get("engage"),
    }
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Champion:", champion)
    print("Dataset:", OUT_DATASET)
    print("Summary:", OUT_SUMMARY)
    print("Sequences:", len(rows), "state_dim", summary["state_dim"])
    print("Action counts:", summary["action_counts"])
    print("Active counts:", summary["active_counts"][:10])
    print("Top pairs:", summary["top_pairs"][:12])


if __name__ == "__main__":
    main()
