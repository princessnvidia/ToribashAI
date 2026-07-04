#!/usr/bin/env python3
"""
evolution_xioi_rpl_hybrid_v32.py

V32 = évolution replay-native :
- parent = .rpl complet qui marche déjà
- on garde POS/QAT/LINVEL/ANGVEL/NEWGAME/ENGAGE intacts
- on mute seulement quelques JOINT après frame 70
- Lua ne contrôle pas le Tori : il score passivement le replay complet

Usage recommandé :
    python3 scripts/evolution_xioi_rpl_hybrid_v32.py generate
    # regarder les replays / ou utiliser le scorer Lua
    python3 scripts/evolution_xioi_rpl_hybrid_v32.py promote xioi_hybrid_v32_g001_c07.rpl
"""
from __future__ import annotations

import json
import random
import re
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
SCRIPTS = ROOT / "scripts"
OUT_DIR = ROOT / "generated_replays"
STATE_DIR = ROOT / "evolution" / "xioi_replay_hybrid_v32"

STEAM_BASE = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
STEAM_REPLAY_DIR = STEAM_BASE / "replay"
STEAM_SCRIPT_DIR = STEAM_BASE / "data" / "script"

SCORER_LUA_NAME = "toribash_xioi_replay_scorer_v32.lua"
PROJECT_LUA = SCRIPTS / SCORER_LUA_NAME
STEAM_LUA = STEAM_SCRIPT_DIR / SCORER_LUA_NAME
STEAM_META = STEAM_SCRIPT_DIR / "xioi_replay_hybrid_v32_current.json"
STEAM_SCORE = STEAM_SCRIPT_DIR / "xioi_replay_hybrid_v32_score.json"

PARENT_CANDIDATES = [
    OUT_DIR / "xioi_master_final_v5_champion.rpl",
    OUT_DIR / "xioi_v30_23_mut.rpl",
    OUT_DIR / "xioi_source_template_v28.rpl",
]
CHAMPION = OUT_DIR / "xioi_replay_hybrid_v32_champion.rpl"
HISTORY = OUT_DIR / "xioi_replay_hybrid_v32_history.jsonl"

POPULATION = 10
DEFAULT_GENERATION = 1
FRAME_LOCK = 70
LIGHT_START = 70
LIGHT_END = 150

JOINT_LINE_RE = re.compile(r"^(JOINT\s+0;\s*)(.*)$")
FRAME_RE = re.compile(r"^FRAME\s+(-?\d+);")

# Mutations d'équilibre légères : épaules/pecs/bras/bassin/jambes, mais très faible amplitude.
BALANCE_JOINTS = {2, 3, 4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19}
CORE_JOINTS = {2, 3, 4, 5, 6, 7}


def ensure_dirs() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STEAM_REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    STEAM_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)


def find_parent() -> Path:
    if CHAMPION.exists():
        return CHAMPION
    for p in PARENT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Aucun parent trouvé. Attendu xioi_master_final_v5_champion.rpl ou xioi_v30_23_mut.rpl")


def current_generation() -> int:
    state_path = STATE_DIR / "state.json"
    if state_path.exists():
        try:
            return int(json.loads(state_path.read_text()).get("generation", 0)) + 1
        except Exception:
            return DEFAULT_GENERATION
    return DEFAULT_GENERATION


def save_generation(gen: int) -> None:
    (STATE_DIR / "state.json").write_text(json.dumps({"generation": gen}, indent=2), encoding="utf-8")


def parse_joint_pairs(text: str) -> dict[int, int]:
    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    pairs: dict[int, int] = {}
    # format attendu : "18 1 19 1" ou répété
    for i in range(0, len(nums) - 1, 2):
        j, v = nums[i], nums[i + 1]
        if 0 <= j <= 19 and 0 <= v <= 4:
            pairs[j] = v
    return pairs


def format_joint_line(prefix: str, pairs: dict[int, int]) -> str:
    body = " ".join(f"{j} {v}" for j, v in sorted(pairs.items()))
    return prefix + body + "\n"


def mutation_rate(frame: int, candidate_idx: int) -> float:
    if frame < FRAME_LOCK:
        return 0.0
    if LIGHT_START <= frame <= LIGHT_END:
        return 0.003 + candidate_idx * 0.00025
    if frame <= 260:
        return 0.008 + candidate_idx * 0.0005
    return 0.004


def maybe_mutate_pairs(pairs: dict[int, int], frame: int, candidate_idx: int) -> tuple[dict[int, int], int]:
    if frame < FRAME_LOCK:
        return pairs, 0
    out = dict(pairs)
    changes = 0
    rate = mutation_rate(frame, candidate_idx)

    # Modifier très légèrement des joints déjà présents.
    for j in list(out.keys()):
        if j not in BALANCE_JOINTS:
            continue
        if random.random() < rate:
            old = out[j]
            # Favoriser un changement d'un seul cran, rarement un resampling.
            if random.random() < 0.85:
                delta = random.choice([-1, 1])
                nv = max(1, min(4, old + delta))
            else:
                nv = random.randint(1, 4)
            if nv != old:
                out[j] = nv
                changes += 1

    # Très rarement ajouter une micro-correction de bassin/épaules après 100.
    if frame >= 100 and random.random() < rate * 0.35:
        j = random.choice(sorted(CORE_JOINTS))
        nv = random.randint(1, 4)
        if out.get(j) != nv:
            out[j] = nv
            changes += 1

    return out, changes


def mutate_rpl(parent: Path, out_path: Path, generation: int, candidate_idx: int) -> dict:
    current_frame = 0
    total_changes = 0
    joint_lines = 0
    lines_out: list[str] = []

    with parent.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m_frame = FRAME_RE.match(line.strip())
            if m_frame:
                current_frame = int(m_frame.group(1))
                lines_out.append(line)
                continue

            m_joint = JOINT_LINE_RE.match(line)
            if m_joint:
                prefix, body = m_joint.group(1), m_joint.group(2)
                pairs = parse_joint_pairs(body)
                if pairs:
                    joint_lines += 1
                    pairs2, ch = maybe_mutate_pairs(pairs, current_frame, candidate_idx)
                    total_changes += ch
                    lines_out.append(format_joint_line(prefix, pairs2))
                else:
                    lines_out.append(line)
                continue

            # Renommer fightname pour ne pas confondre dans Toribash.
            if line.startswith("FIGHTNAME 0;"):
                lines_out.append(f"FIGHTNAME 0; xioi_hybrid_v32_g{generation:03d}_c{candidate_idx:02d}\n")
            else:
                lines_out.append(line)

    out_path.write_text("".join(lines_out), encoding="utf-8")
    return {"candidate": candidate_idx, "changes": total_changes, "joint_lines": joint_lines, "path": str(out_path)}


def clean_old_lua() -> None:
    keep = {
        "toribash_upright_runner_v18.lua",
        "toribash_recovery_runner_v1.lua",
        SCORER_LUA_NAME,
    }
    for p in STEAM_SCRIPT_DIR.glob("*.lua"):
        if p.name not in keep:
            try:
                p.unlink()
            except Exception:
                pass


def deploy_lua() -> None:
    if not PROJECT_LUA.exists():
        raise FileNotFoundError(PROJECT_LUA)
    shutil.copy2(PROJECT_LUA, STEAM_LUA)


def write_meta(gen: int, cand: int) -> None:
    data = {"version": 32, "generation": gen, "candidate": cand, "population": POPULATION}
    STEAM_META.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if STEAM_SCORE.exists():
        STEAM_SCORE.unlink()


def generate() -> None:
    ensure_dirs()
    clean_old_lua()
    deploy_lua()

    parent = find_parent()
    gen = current_generation()
    print("Parent:", parent)
    print("Generation:", gen)
    print("Population:", POPULATION)
    print("Lua scorer:", STEAM_LUA)

    results = []
    # candidat 0 = parent exact, pour toujours garder un témoin sain.
    parent_copy = OUT_DIR / f"xioi_hybrid_v32_g{gen:03d}_c00_PARENT.rpl"
    shutil.copy2(parent, parent_copy)
    shutil.copy2(parent_copy, STEAM_REPLAY_DIR / parent_copy.name)
    results.append({"candidate": 0, "changes": 0, "joint_lines": None, "path": str(parent_copy), "parent": True})

    for idx in range(1, POPULATION + 1):
        out = OUT_DIR / f"xioi_hybrid_v32_g{gen:03d}_c{idx:02d}.rpl"
        info = mutate_rpl(parent, out, gen, idx)
        shutil.copy2(out, STEAM_REPLAY_DIR / out.name)
        results.append(info)

    save_generation(gen)
    manifest = {
        "version": 32,
        "generation": gen,
        "population": POPULATION,
        "parent": str(parent),
        "candidates": results,
        "note": "Play candidates in Toribash with scorer Lua loaded. Promote the best manually.",
    }
    (OUT_DIR / f"xioi_hybrid_v32_g{gen:03d}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Generated candidates copied to Steam replay dir.")
    print("Load scorer in Toribash:")
    print(f"  /ls {SCORER_LUA_NAME}")
    print("Then test replays xioi_hybrid_v32_g%03d_*" % gen)
    print("When testing candidate N, optional meta update:")
    print("  python3 scripts/evolution_xioi_rpl_hybrid_v32.py meta %d N" % gen)


def promote(name: str) -> None:
    ensure_dirs()
    src = Path(name)
    if not src.exists():
        src = OUT_DIR / name
    if not src.exists():
        src = STEAM_REPLAY_DIR / name
    if not src.exists():
        raise FileNotFoundError(name)

    shutil.copy2(src, CHAMPION)
    shutil.copy2(src, STEAM_REPLAY_DIR / CHAMPION.name)
    event = {"time": time.time(), "promoted": str(src), "champion": str(CHAMPION)}
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
    print("Promoted:", src)
    print("Champion:", CHAMPION)
    print("Copied to Steam replay dir.")


def meta(gen: int, cand: int) -> None:
    ensure_dirs()
    write_meta(gen, cand)
    print("Meta written:", STEAM_META)


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "generate"
    if cmd == "generate":
        generate()
    elif cmd == "promote":
        if len(sys.argv) < 3:
            raise SystemExit("Usage: evolution_xioi_rpl_hybrid_v32.py promote <replay_name>")
        promote(sys.argv[2])
    elif cmd == "meta":
        if len(sys.argv) < 4:
            raise SystemExit("Usage: evolution_xioi_rpl_hybrid_v32.py meta <generation> <candidate>")
        meta(int(sys.argv[2]), int(sys.argv[3]))
    else:
        raise SystemExit(f"Unknown command: {cmd}")


if __name__ == "__main__":
    main()
