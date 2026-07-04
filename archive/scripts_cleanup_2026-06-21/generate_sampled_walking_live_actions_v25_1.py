#!/usr/bin/env python3
"""
generate_sampled_walking_live_actions_v25_1.py

V25.1 = même expérience que V25, mais sans parser JSON côté Lua.
Python génère directement une table Lua:

  curated_walking_sampled_v25_1_actions_table.lua

Le runner fait dofile() et reçoit ACTIONS_BY_FRAME / ACTION_FRAMES.
But: éviter le bug V25 où le parser Lua ne lisait qu'une seule action.
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
OUT_JSON = OUT_DIR / "curated_walking_sampled_v25_1_live_actions.json"
OUT_LUA_TABLE = OUT_DIR / "curated_walking_sampled_v25_1_actions_table.lua"

TORIBASH_SCRIPT_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"
)
TORIBASH_JSON = TORIBASH_SCRIPT_DIR / "curated_walking_sampled_v25_1_live_actions_current.json"
TORIBASH_LUA_TABLE = TORIBASH_SCRIPT_DIR / "curated_walking_sampled_v25_1_actions_table.lua"

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
    activity_score = -abs(avg_active - 4.0) * 4.0
    return len(rows) * 0.7 + min(span, 260) * 0.08 + activity_score + bonus


def choose_groups(groups: dict[str, list[dict[str, Any]]]) -> list[tuple[str, list[dict[str, Any]], float]]:
    scored = [(src, rows, group_score(src, rows)) for src, rows in groups.items()]
    scored = [x for x in scored if x[2] > -999]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def make_action_stream(selected: list[tuple[str, list[dict[str, Any]], float]]) -> tuple[list[dict[str, Any]], list[str]]:
    actions: list[dict[str, Any]] = []
    used_sources: list[str] = []
    if not selected:
        raise RuntimeError("Aucun groupe exploitable dans le dataset curated")

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


def lua_quote(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def write_lua_table(path: Path, actions: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("-- curated_walking_sampled_v25_1_actions_table.lua")
    lines.append("-- Auto-generated by generate_sampled_walking_live_actions_v25_1.py")
    lines.append("CURATED_WALKING_V25_1_META = {")
    lines.append(f"  name = {lua_quote(str(metadata.get('name', 'curated_walking_sampled_v25_1')))},")
    lines.append(f"  version = {lua_quote(str(metadata.get('version', '25.1')))},")
    lines.append(f"  turnframes = {int(metadata.get('turnframes', TURNFRAMES))},")
    lines.append(f"  generated_steps = {len(actions)},")
    lines.append("}")
    lines.append("")
    lines.append("ACTIONS_BY_FRAME = {")
    for action in actions:
        frame = int(action["frame"])
        pairs = action.get("pairs", [])
        lines.append(f"  [{frame}] = {{")
        for j, v in pairs:
            lines.append(f"    {{ {int(j)}, {int(v)} }},")
        lines.append("  },")
    lines.append("}")
    lines.append("")
    lines.append("ACTION_FRAMES = {")
    for action in actions:
        lines.append(f"  {int(action['frame'])},")
    lines.append("}")
    lines.append("")
    lines.append("return { meta = CURATED_WALKING_V25_1_META, actions = ACTIONS_BY_FRAME, frames = ACTION_FRAMES }")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "name": "curated_walking_sampled_v25_1_live_actions",
        "version": "25.1",
        "mode": "sample_real_curated_dataset_actions_lua_table_no_json_parser",
        "dataset": str(DATASET_PATH),
        "turnframes": TURNFRAMES,
        "generated_steps": len(actions),
        "selected_sources": [src for src, _, _ in selected[:4]],
        "active_histogram": dict(sorted(active_counter.items())),
        "top_pairs": [[[j, v], c] for (j, v), c in pair_counter.most_common(30)],
        "actions": actions,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    write_lua_table(OUT_LUA_TABLE, actions, data)

    TORIBASH_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_JSON, TORIBASH_JSON)
    shutil.copy2(OUT_LUA_TABLE, TORIBASH_LUA_TABLE)

    print("JSON projet:", OUT_JSON)
    print("Lua table projet:", OUT_LUA_TABLE)
    print("JSON Steam:", TORIBASH_JSON)
    print("Lua table Steam:", TORIBASH_LUA_TABLE)
    print("Active histogram:", dict(sorted(active_counter.items())))
    print("Top pairs:", pair_counter.most_common(15))
    print("First 5 actions:")
    for a in actions[:5]:
        print(" ", a)


if __name__ == "__main__":
    main()
