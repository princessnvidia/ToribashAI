#!/usr/bin/env python3
"""
evolution_xioi_seed_v27.py

Étape 4 / V27:
  Crée une petite population de variations autour du seed Xioi sample/teacher.

Cette V27 ne dépend pas encore d'un reward live automatique: elle génère des tables Lua candidates
à tester rapidement. Ensuite on pourra brancher un vrai runner reward si une candidate est prometteuse.

Sorties:
  generated_replays/xioi_v27_evo_candidates/*.json
  data/script/xioi_v27_live_actions_table.lua (candidate 00 par défaut)
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET_PATH = ROOT / "datasets" / "ml" / "xioi_only_v26_sequences.jsonl"
OUT_DIR = ROOT / "generated_replays" / "xioi_v27_evo_candidates"
TORIBASH_SCRIPT_DIR = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"
OUT_LUA_TABLE = TORIBASH_SCRIPT_DIR / "xioi_v27_live_actions_table.lua"

SEED = 2704
POPULATION = 12
STEPS = 220
TURNFRAMES = 5
MUTATE_RATE = 0.08
DROP_RATE = 0.03
ADD_RATE = 0.04
JOINTS = list(range(20))
VALUES = [1, 2, 3, 4]

random.seed(SEED)


def load_rows() -> list[dict[str, Any]]:
    rows = []
    with DATASET_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError("Dataset vide")
    return sorted(rows, key=lambda r: int(r.get("target_frame", 0)))


def action_to_pairs(action: list[int]) -> list[list[int]]:
    return [[j, int(v)] for j, v in enumerate(action) if int(v) != 0]


def base_actions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for step in range(STEPS):
        r = rows[step % len(rows)]
        out.append({
            "frame": step * TURNFRAMES,
            "pairs": action_to_pairs(r["action"]),
            "source_frame": r.get("target_frame"),
        })
    return out


def mutate_pairs(pairs: list[list[int]], generation_bias: float = 1.0) -> list[list[int]]:
    d = {int(j): int(v) for j, v in pairs}
    # Mutations douces: on évite de tout casser.
    for j in list(d.keys()):
        if random.random() < DROP_RATE * generation_bias:
            del d[j]
            continue
        if random.random() < MUTATE_RATE * generation_bias:
            d[j] = random.choice(VALUES)
    if random.random() < ADD_RATE * generation_bias and len(d) < 12:
        d[random.choice(JOINTS)] = random.choice(VALUES)
    return [[j, d[j]] for j in sorted(d.keys())]


def mutate_actions(actions: list[dict[str, Any]], idx: int) -> list[dict[str, Any]]:
    bias = 0.5 + idx / max(1, POPULATION - 1)
    out = []
    for a in actions:
        pairs = mutate_pairs(a.get("pairs", []), generation_bias=bias)
        out.append({
            "frame": a["frame"],
            "pairs": pairs,
            "source_frame": a.get("source_frame"),
            "candidate": idx,
        })
    return out


def write_lua_table(actions: list[dict[str, Any]]) -> None:
    TORIBASH_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    lines = ["-- Auto-generated candidate by evolution_xioi_seed_v27.py", "XIOI_V27_ACTIONS = {"]
    for a in actions:
        pairs = ", ".join("{" + str(int(j)) + "," + str(int(v)) + "}" for j, v in a.get("pairs", []))
        lines.append(f"  [{int(a['frame'])}] = {{ {pairs} }},")
    lines.extend(["}", f"XIOI_V27_ACTION_COUNT = {len(actions)}", f"XIOI_V27_TURNFRAMES = {TURNFRAMES}"])
    OUT_LUA_TABLE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_rows()
    base = base_actions(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    candidates = []

    for i in range(POPULATION):
        actions = base if i == 0 else mutate_actions(base, i)
        data = {
            "version": 27,
            "name": f"xioi_v27_candidate_{i:02d}",
            "turnframes": TURNFRAMES,
            "steps": STEPS,
            "mutation": "none" if i == 0 else "soft_joint_mutation",
            "actions": actions,
        }
        path = OUT_DIR / f"xioi_v27_candidate_{i:02d}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        candidates.append(path)
        print("made", path)

    write_lua_table(base)
    print("Default table = candidate 00 seed:", OUT_LUA_TABLE)
    print("Pour tester une autre candidate, on ajoutera un petit loader/copy dans V27.1.")


if __name__ == "__main__":
    main()
