#!/usr/bin/env python3
"""
build_curated_walking_dataset_v23_1.py

V23.1 = curated walking dataset depuis les replays que Vio a repérés / candidats proches.
But: ne plus matcher trop strictement les noms. On prend une liste explicite de fichiers si présents,
puis quelques mots-clés larges. On extrait surtout le début du replay.

Sorties:
  datasets/ml/curated_walking_v23_1_sequences.jsonl
  generated_replays/curated_walking_v23_1_summary.json
  generated_replays/curated_walking_v23_1_sources.json
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
PARKOUR_JSON = ROOT / "datasets" / "parkour_json"
OUT_DATASET = ROOT / "datasets" / "ml" / "curated_walking_v23_1_sequences.jsonl"
OUT_SUMMARY = ROOT / "generated_replays" / "curated_walking_v23_1_summary.json"
OUT_SOURCES = ROOT / "generated_replays" / "curated_walking_v23_1_sources.json"

SEQ_LEN = 8
MAX_FRAME = 260
MIN_ACTIONS = 8

# Fichiers trouvés par Vio avec recherche large.
EXPLICIT_FILE_SUBSTRINGS = [
    "The Kurr Of Treasure-hunting",
    "Swex- Divine",
    "pakourxioi",
    "[raid v41-2]",
    "_Xioi_Pk - Budokai",
    "_Xioi_Pk - Ykeus",
    "_Xioi- PK- Ravine Explorer",
    "[raid v41]",
    "Raid Challenge",
    "raid challenge",
    "[P - A] another tomb raid final",
    "Karbn- run away",
    "Flash parkour Karbn",
]

# Match large en backup, mais on évite de prendre tout karbn/xioi si ça explose trop.
KEYWORDS = [
    "xioi", "swex", "divine", "pakourxioi", "raid", "ravine", "treasure", "karbn", "run away", "flash parkour"
]

JOINT_DIM = 20


def norm(s: str) -> str:
    return s.lower().replace("_", " ").replace("-", " ").replace(".", " ")


def select_files() -> list[Path]:
    all_files = sorted(PARKOUR_JSON.glob("*.json"))
    selected: list[Path] = []
    seen = set()

    for sub in EXPLICIT_FILE_SUBSTRINGS:
        nsub = norm(sub)
        for p in all_files:
            if p in seen:
                continue
            if nsub in norm(p.name):
                selected.append(p)
                seen.add(p)

    # backup keyword matches, capped to avoid returning the whole parkour set.
    for p in all_files:
        if p in seen:
            continue
        np = norm(p.name)
        if any(k in np for k in KEYWORDS):
            selected.append(p)
            seen.add(p)

    return selected


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        print("BAD JSON", path.name, e)
        return None


def sorted_frames(data: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    frames = data.get("frames", {})
    if isinstance(frames, dict):
        out = []
        for k, v in frames.items():
            try:
                out.append((int(k), v))
            except Exception:
                pass
        return sorted(out, key=lambda x: x[0])
    if isinstance(frames, list):
        return [(i, f) for i, f in enumerate(frames) if isinstance(f, dict)]
    return []


def p0(frame: dict[str, Any]) -> dict[str, Any]:
    return frame.get("players", {}).get("0", {})


def get_pairs(frame: dict[str, Any]) -> list[list[int]]:
    player = p0(frame)
    pairs = player.get("joint_pairs") or []
    clean = []
    for pair in pairs:
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            try:
                j = int(pair[0]); v = int(pair[1])
                if 0 <= j < JOINT_DIM and 0 <= v <= 4:
                    clean.append([j, v])
            except Exception:
                pass
    return clean


def get_pos(frame: dict[str, Any]) -> list[list[float]]:
    pos = p0(frame).get("pos") or []
    out = []
    for item in pos:
        if isinstance(item, (list, tuple)) and len(item) >= 3:
            try:
                out.append([float(item[0]), float(item[1]), float(item[2])])
            except Exception:
                out.append([0.0, 0.0, 0.0])
    return out


def vec_features(pos: list[list[float]]) -> list[float]:
    # 42 dims = 14 body points * xyz. Identique à l'esprit V23.
    # On prend les 14 premiers points stables si disponibles.
    feats: list[float] = []
    for i in range(14):
        if i < len(pos):
            feats.extend(pos[i][:3])
        else:
            feats.extend([0.0, 0.0, 0.0])
    return feats


def action_vector(pairs: list[list[int]]) -> list[int]:
    a = [0] * JOINT_DIM
    for j, v in pairs:
        a[j] = int(v)
    return a


def usable_frame(frame_no: int, frame: dict[str, Any]) -> bool:
    if frame_no > MAX_FRAME:
        return False
    if not get_pos(frame):
        return False
    return True


def main() -> None:
    paths = select_files()
    print("Curated V23.1 files:", len(paths))
    for p in paths:
        print(" -", p.name)

    OUT_DATASET.parent.mkdir(parents=True, exist_ok=True)
    OUT_SUMMARY.parent.mkdir(parents=True, exist_ok=True)

    total_sequences = 0
    action_counts = Counter()
    source_summary = []

    with OUT_DATASET.open("w", encoding="utf-8") as out:
        for path in paths:
            data = load_json(path)
            if not data:
                continue
            frames = [(fn, fr) for fn, fr in sorted_frames(data) if usable_frame(fn, fr)]
            if len(frames) < SEQ_LEN + 1:
                source_summary.append({"file": path.name, "usable": len(frames), "sequences": 0})
                continue

            rows = []
            for fn, fr in frames:
                pos = get_pos(fr)
                pairs = get_pairs(fr)
                rows.append({
                    "frame": fn,
                    "state": vec_features(pos),
                    "pairs": pairs,
                    "action": action_vector(pairs),
                })

            seq_count = 0
            for i in range(0, len(rows) - SEQ_LEN):
                # Target = next visible action after sequence.
                target = rows[i + SEQ_LEN]
                if not target["pairs"]:
                    continue
                state_seq = [r["state"] for r in rows[i:i+SEQ_LEN]]
                item = {
                    "version": "23.1",
                    "source_file": path.name,
                    "start_frame": rows[i]["frame"],
                    "target_frame": target["frame"],
                    "state_seq": state_seq,
                    "action": target["action"],
                    "pairs": target["pairs"],
                }
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
                seq_count += 1
                total_sequences += 1
                action_counts.update(target["action"])

            first = frames[0][0] if frames else None
            last = frames[-1][0] if frames else None
            source_summary.append({
                "file": path.name,
                "usable": len(frames),
                "sequences": seq_count,
                "frame_min": first,
                "frame_max": last,
            })
            print(f"{path.name}: usable={len(frames)} sequences={seq_count} frames={first}-{last}")

    summary = {
        "version": "23.1",
        "selected_files": len(paths),
        "sequences": total_sequences,
        "seq_len": SEQ_LEN,
        "state_dim": 42,
        "action_dim": JOINT_DIM,
        "max_frame": MAX_FRAME,
        "action_counts": action_counts.most_common(),
        "sources": source_summary,
    }
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_SOURCES.write_text(json.dumps({"files": source_summary}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSaved dataset:", OUT_DATASET)
    print("Saved summary:", OUT_SUMMARY)
    print("Sequences:", total_sequences)
    print("Action counts:", action_counts.most_common())


if __name__ == "__main__":
    main()
