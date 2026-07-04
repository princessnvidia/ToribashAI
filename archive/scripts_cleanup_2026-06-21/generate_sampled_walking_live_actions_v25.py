#!/usr/bin/env python3
"""
generate_sampled_walking_live_actions_v25.py

V25 = on arrête la génération GRU en boucle ouverte qui collapse en pose fixe.
Ici on sample/rejoue directement les vraies actions du dataset curated walking V23.1.
But: vérifier que le runner live + dataset curated produisent une vraie séquence de marche.

Entrée:
  datasets/ml/curated_walking_v23_1_sequences.jsonl

Sorties:
  generated_replays/curated_walking_sampled_v25_live_actions.json
  data/script/curated_walking_sampled_v25_live_actions_current.json
"""

from __future__ import annotations

import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET_PATH = ROOT / "datasets" / "ml" / "curated_walking_v23_1_sequences.jsonl"
OUT_DIR = ROOT / "generated_replays"
OUT_ACTIONS = OUT_DIR / "curated_walking_sampled_v25_live_actions.json"
TORIBASH_SCRIPT_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"
)
TORIBASH_ACTIONS = TORIBASH_SCRIPT_DIR / "curated_walking_sampled_v25_live_actions_current.json"

TURNFRAMES = 5
TARGET_STEPS = 180
MIN_GROUP_ROWS = 24
RANDOM_SEED = 2501


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"Dataset vide: {path}")
    return rows


def get_source(row: dict[str, Any]) -> str:
    for key in ("source", "replay", "file", "filename"):
        v = row.get(key)
        if isinstance(v, str) and v:
            return Path(v).name
    return "unknown_source"


def get_frame(row: dict[str, Any]) -> int:
    for key in ("target_frame", "frame", "frame_no", "target"):
        v = row.get(key)
        if isinstance(v, (int, float, str)):
            try:
                return int(v)
            except Exception:
                pass
    return 0


def get_action(row: dict[str, Any]) -> list[int] | None:
    # Dataset V23.1 stores a 20-value target action. Keep this robust.
    for key in ("action", "target_action", "y", "target", "actions"):
        v = row.get(key)
        if isinstance(v, list) and len(v) == 20:
            return [int(x) for x in v]
    return None


def action_to_pairs(action: list[int]) -> list[list[int]]:
    return [[int(j), int(v)] for j, v in enumerate(action) if int(v) != 0]


def active_count(row: dict[str, Any]) -> int:
    a = get_action(row)
    return sum(1 for v in (a or []) if int(v) != 0)


def build_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if get_action(row) is None:
            continue
        groups[get_source(row)].append(row)
    for src in list(groups):
        groups[src].sort(key=get_frame)
    return groups


def group_score(src: str, rows: list[dict[str, Any]]) -> float:
    # Prefer enough frames, moderate activity, and curated walking-looking names.
    if len(rows) < MIN_GROUP_ROWS:
        return -9999.0
    counts = [active_count(r) for r in rows]
    avg_active = sum(counts) / max(1, len(counts))
    frames = [get_frame(r) for r in rows]
    span = max(frames) - min(frames) if frames else 0
    name = src.lower()
    bonus = 0.0
    for token in ("xioi", "walk", "run", "karbn", "kurr", "swex", "raid"):
        if token in name:
            bonus += 8.0
    # Too much activity usually equals trick/explosion; too little equals frozen.
    activity_score = -abs(avg_active - 4.0) * 4.0
    return len(rows) * 0.7 + min(span, 260) * 0.08 + activity_score + bonus


def choose_groups(groups: dict[str, list[dict[str, Any]]]) -> list[tuple[str, list[dict[str, Any]], float]]:
    scored = [(src, rows, group_score(src, rows)) for src, rows in groups.items()]
    scored = [x for x in scored if x[2] > -999]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def make_action_stream(selected: list[tuple[str, list[dict[str, Any]], float]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Build a long action stream from real curated rows.

    We do not call the GRU here. This is a control experiment: if this produces
    visible movement, the live runner and curated dataset are usable. Then V26
    can train/generate with better feedback.
    """
    actions: list[dict[str, Any]] = []
    used_sources: list[str] = []

    if not selected:
        raise RuntimeError("Aucun groupe exploitable dans le dataset curated")

    # Use top 4 sources in small blocks so the motion has real variation.
    top = selected[:4]
    cursors = {src: 0 for src, _, _ in top}
    block_len = 24
    step = 0

    while len(actions) < TARGET_STEPS:
        for src, rows, _score in top:
            used_sources.append(src)
            start = cursors[src]
            end = min(len(rows), start + block_len)
            if end - start < 4:
                start = 0
                end = min(len(rows), block_len)
            cursors[src] = end

            for row in rows[start:end]:
                a = get_action(row)
                if a is None:
                    continue
                pairs = action_to_pairs(a)
                actions.append({
                    "frame": step * TURNFRAMES,
                    "pairs": pairs,
                    "source": src,
                    "source_frame": get_frame(row),
                    "active": len(pairs),
                })
                step += 1
                if len(actions) >= TARGET_STEPS:
                    break
            if len(actions) >= TARGET_STEPS:
                break

    return actions[:TARGET_STEPS], used_sources


def main() -> None:
    random.seed(RANDOM_SEED)
    rows = load_rows(DATASET_PATH)
    groups = build_groups(rows)
    selected = choose_groups(groups)

    print("Dataset:", DATASET_PATH)
    print("Rows:", len(rows))
    print("Groups:", len(groups))
    print("Top groups:")
    for src, gr, sc in selected[:12]:
        counts = [active_count(r) for r in gr]
        frames = [get_frame(r) for r in gr]
        print(
            f"  score={sc:7.2f} rows={len(gr):4d} "
            f"frames={min(frames) if frames else 0}-{max(frames) if frames else 0} "
            f"avg_active={sum(counts)/max(1,len(counts)):.2f} src={src}"
        )

    actions, used_sources = make_action_stream(selected)

    active_counter = Counter(len(a["pairs"]) for a in actions)
    pair_counter: Counter[tuple[int, int]] = Counter()
    for a in actions:
        for j, v in a["pairs"]:
            pair_counter[(int(j), int(v))] += 1

    data = {
        "name": "curated_walking_sampled_v25_live_actions",
        "version": "25",
        "mode": "sample_real_curated_dataset_actions_no_gru",
        "dataset": str(DATASET_PATH),
        "turnframes": TURNFRAMES,
        "generated_steps": len(actions),
        "selected_sources": [src for src, _, _ in selected[:4]],
        "active_histogram": dict(sorted(active_counter.items())),
        "top_pairs": [[[j, v], c] for (j, v), c in pair_counter.most_common(30)],
        "actions": actions,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ACTIONS.write_text(json.dumps(data, indent=2), encoding="utf-8")

    TORIBASH_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_ACTIONS, TORIBASH_ACTIONS)

    print("Actions projet:", OUT_ACTIONS)
    print("Actions Steam:", TORIBASH_ACTIONS)
    print("Active histogram:", dict(sorted(active_counter.items())))
    print("Top pairs:", pair_counter.most_common(15))
    print("First 5 actions:")
    for a in actions[:5]:
        print(" ", a)


if __name__ == "__main__":
    main()
