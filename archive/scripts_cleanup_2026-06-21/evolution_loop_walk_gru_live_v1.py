#!/usr/bin/env python3
"""
evolution_loop_walk_gru_live_v1.py

ToribashAI live evolution branch from the GRU/template-safe walking seed.

Important:
- This is NOT RPL replay evolution anymore.
- Python exports a live agent to data/script/toribash_walk_gru_live_agent_v1.lua.
- Lua applies that agent in Toribash, auto-starts, scores, and writes a JSON result.
- The first 0..315 frames are protected because they are the launch / entry into walking.
- After 315, mutations are tiny around the learned len265 GRU loop.

Typical use:
  cp ~/Téléchargements/*walk_gru_live_v1* ~/Documents/ToribashAI/scripts/
  cd ~/Documents/ToribashAI
  python3 scripts/evolution_loop_walk_gru_live_v1.py setup
  # in Toribash once: /ls toribash_walk_gru_live_runner_v1.lua
  python3 scripts/evolution_loop_walk_gru_live_v1.py run

If automation is unreliable:
  python3 scripts/evolution_loop_walk_gru_live_v1.py candidate 0
  # then manually /reset and /ls toribash_walk_gru_live_runner_v1.lua in Toribash
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

ROOT = Path.home() / "Documents/ToribashAI"
SCRIPTS = ROOT / "scripts"
GEN = ROOT / "generated_replays"
EVOL = ROOT / "evolution" / "walk_gru_live_v1"
POP = EVOL / "population"
BEST = EVOL / "best"
STATE = EVOL / "state.json"

STEAM = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
STEAM_SCRIPT = STEAM / "data" / "script"

LUA_NAME = "toribash_walk_gru_live_runner_v1.lua"
PROJECT_LUA = SCRIPTS / LUA_NAME
STEAM_LUA = STEAM_SCRIPT / LUA_NAME

AGENT_LUA = STEAM_SCRIPT / "toribash_walk_gru_live_agent_v1.lua"
AGENT_JSON = STEAM_SCRIPT / "toribash_walk_gru_live_agent_v1.json"
META_TXT = STEAM_SCRIPT / "toribash_walk_gru_live_meta_v1.txt"
RESULT_JSON = STEAM_SCRIPT / "toribash_walk_gru_live_result_v1.json"

# Best known RPL seeds. We parse JOINT commands from one of these.
SEED_RPL_CANDIDATES = [
    GEN / "xioi_loop_len265_gru_long_v54_seed048_template_safe.rpl",
    GEN / "xioi_loop_len265_gru_long_v54_seed008_template_safe.rpl",
    GEN / "xioi_loop_len265_gru_v53_free_template_safe.rpl",
    GEN / "xioi_loop_len265_gru_v53_teacher_template_safe.rpl",
    GEN / "xioi_loop_len265_champion_v51.rpl",
    GEN / "xioi_loop_phase_v50_len265.rpl",
]

PROTECT_UNTIL = 315
POPULATION = 10
MAX_FRAME = 1100
RUN_TIMEOUT = 35.0
RANDOM_SEED = 551001

# Mutation schedule: intentionally gentle. We already have a walking loop.
MUTATION_SCHEDULE = [
    # min_frame, max_frame, mutate_existing_rate, add_pair_rate, drop_pair_rate
    (316, 500, 0.006, 0.004, 0.002),
    (501, 800, 0.010, 0.006, 0.003),
    (801, 99999, 0.014, 0.010, 0.004),
]
BALANCE_JOINTS = [1, 2, 3, 4, 7, 12, 13, 14, 15, 16, 17, 18, 19]
ALL_JOINTS = list(range(20))

FRAME_RE = re.compile(r"^FRAME\s+(\d+);")
JOINT_RE = re.compile(r"^JOINT\s+0;\s*(.*)$")

Command = Dict[str, object]
Agent = Dict[str, object]


def ensure_dirs() -> None:
    for p in (EVOL, POP, BEST, STEAM_SCRIPT):
        p.mkdir(parents=True, exist_ok=True)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"generation": 1, "best_score": None, "best_agent": None, "history": []}


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def find_seed_rpl() -> Path:
    for p in SEED_RPL_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("No walking seed RPL found. Expected one of:\n" + "\n".join(str(p) for p in SEED_RPL_CANDIDATES))


def parse_pairs(text: str) -> List[List[int]]:
    nums = [int(x) for x in re.findall(r"-?\d+", text)]
    pairs: List[List[int]] = []
    for i in range(0, len(nums) - 1, 2):
        j, v = nums[i], nums[i + 1]
        if 0 <= j < 20 and 0 <= v <= 4 and v != 0:
            pairs.append([j, v])
    # de-duplicate per frame, last value wins
    d: Dict[int, int] = {}
    for j, v in pairs:
        d[j] = v
    return [[j, d[j]] for j in sorted(d)]


def parse_rpl_actions(path: Path) -> List[Command]:
    actions_by_frame: Dict[int, Dict[int, int]] = {}
    current_frame: Optional[int] = None
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = FRAME_RE.match(line)
        if m:
            current_frame = int(m.group(1))
            continue
        jm = JOINT_RE.match(line)
        if jm and current_frame is not None:
            for j, v in parse_pairs(jm.group(1)):
                actions_by_frame.setdefault(current_frame, {})[j] = v
    out: List[Command] = []
    for fr in sorted(actions_by_frame):
        if fr > MAX_FRAME:
            continue
        pairs = [[j, actions_by_frame[fr][j]] for j in sorted(actions_by_frame[fr])]
        out.append({"frame": fr, "pairs": pairs})
    if not out:
        raise RuntimeError(f"No JOINT 0 actions parsed from {path}")
    return out


def make_seed_agent() -> Agent:
    seed = find_seed_rpl()
    actions = parse_rpl_actions(seed)
    return {
        "name": "walk_gru_live_seed_v1",
        "version": "walk_gru_live_v1",
        "source_rpl": str(seed),
        "generation": 0,
        "agent_id": "seed",
        "population": POPULATION,
        "protect_until": PROTECT_UNTIL,
        "max_frame": MAX_FRAME,
        "commands": actions,
        "parent_score": None,
    }


def load_best_or_seed() -> Agent:
    state = load_state()
    best_path = state.get("best_agent")
    if best_path and Path(best_path).exists():
        return json.loads(Path(best_path).read_text(encoding="utf-8"))
    seed_path = BEST / "walk_gru_live_seed_v1.json"
    if seed_path.exists():
        return json.loads(seed_path.read_text(encoding="utf-8"))
    agent = make_seed_agent()
    seed_path.write_text(json.dumps(agent, indent=2), encoding="utf-8")
    state["best_agent"] = str(seed_path)
    state["best_score"] = None
    save_state(state)
    return agent


def rate_for_frame(frame: int) -> Tuple[float, float, float]:
    if frame <= PROTECT_UNTIL:
        return 0.0, 0.0, 0.0
    for lo, hi, m, a, d in MUTATION_SCHEDULE:
        if lo <= frame <= hi:
            return m, a, d
    return 0.0, 0.0, 0.0


def mutate_agent(parent: Agent, generation: int, agent_idx: int) -> Agent:
    child = json.loads(json.dumps(parent))
    child["name"] = f"walk_gru_live_v1_g{generation:04d}_a{agent_idx:02d}"
    child["generation"] = generation
    child["agent_id"] = agent_idx
    child["population"] = POPULATION
    child["parent_score"] = parent.get("parent_score")
    child["protect_until"] = PROTECT_UNTIL
    child["max_frame"] = MAX_FRAME

    mutations = 0
    commands = child.get("commands", [])
    for cmd in commands:
        frame = int(cmd.get("frame", 0))
        mutate_rate, add_rate, drop_rate = rate_for_frame(frame)
        if mutate_rate <= 0:
            continue

        pairs = [[int(j), int(v)] for j, v in cmd.get("pairs", [])]
        d = {j: v for j, v in pairs}

        # Drop only after the protected opening and only rarely.
        for j in list(d.keys()):
            if random.random() < drop_rate:
                del d[j]
                mutations += 1

        # Mutate existing pairs by nudging state ±1, not randomizing violently.
        for j in list(d.keys()):
            if random.random() < mutate_rate:
                old = d[j]
                step = random.choice([-1, 1])
                new = max(1, min(4, old + step))
                if new != old:
                    d[j] = new
                    mutations += 1

        # Add mostly balance/core/hips/shoulders/legs joints.
        if random.random() < add_rate:
            j = random.choice(BALANCE_JOINTS)
            if j not in d:
                d[j] = random.randint(1, 4)
                mutations += 1

        cmd["pairs"] = [[j, d[j]] for j in sorted(d)]

    child["mutation_count"] = mutations
    return child


def lua_quote(s: str) -> str:
    return json.dumps(str(s), ensure_ascii=False)


def export_agent_lua(agent: Agent, path: Path) -> None:
    lines = []
    lines.append("return {")
    lines.append(f"  name = {lua_quote(agent.get('name', 'walk_gru_live'))},")
    lines.append(f"  version = {lua_quote(agent.get('version', 'walk_gru_live_v1'))},")
    lines.append(f"  gen = {int(agent.get('generation', 0))},")
    # agent_id can be string for seed
    aid = agent.get("agent_id", "?")
    if isinstance(aid, int):
        lines.append(f"  agent_id = {aid},")
    else:
        lines.append(f"  agent_id = {lua_quote(str(aid))},")
    lines.append(f"  population = {int(agent.get('population', POPULATION))},")
    lines.append(f"  max_frame = {int(agent.get('max_frame', MAX_FRAME))},")
    lines.append(f"  protect_until = {int(agent.get('protect_until', PROTECT_UNTIL))},")
    lines.append("  actions = {")
    for cmd in sorted(agent.get("commands", []), key=lambda c: int(c.get("frame", 0))):
        pairs = cmd.get("pairs", [])
        pair_txt = ", ".join("{%d,%d}" % (int(j), int(v)) for j, v in pairs)
        lines.append(f"    {{ frame = {int(cmd.get('frame', 0))}, pairs = {{ {pair_txt} }} }},")
    lines.append("  }")
    lines.append("}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_current_agent(agent: Agent, generation: int, agent_idx: int) -> None:
    STEAM_SCRIPT.mkdir(parents=True, exist_ok=True)
    export_agent_lua(agent, AGENT_LUA)
    AGENT_JSON.write_text(json.dumps(agent, indent=2), encoding="utf-8")
    META_TXT.write_text(
        f"gen={generation}\nagent={agent_idx}\npopulation={POPULATION}\nparent_score={agent.get('parent_score')}\nname={agent.get('name')}\n",
        encoding="utf-8",
    )


def setup() -> None:
    ensure_dirs()
    if not PROJECT_LUA.exists():
        raise FileNotFoundError(f"Missing Lua in scripts: {PROJECT_LUA}\nCopy toribash_walk_gru_live_runner_v1.lua into scripts first.")
    shutil.copy2(PROJECT_LUA, STEAM_LUA)
    seed = make_seed_agent()
    seed_path = BEST / "walk_gru_live_seed_v1.json"
    seed_path.write_text(json.dumps(seed, indent=2), encoding="utf-8")
    export_current_agent(seed, 0, 0)
    state = load_state()
    if not state.get("best_agent"):
        state["best_agent"] = str(seed_path)
        state["best_score"] = None
        save_state(state)
    print("Lua copied:", STEAM_LUA)
    print("Seed actions:", len(seed["commands"]))
    print("Seed source:", seed.get("source_rpl"))
    print("Agent exported:", AGENT_LUA)


def clean_lua() -> None:
    keep = {
        "toribash_upright_runner_v18.lua",
        "toribash_recovery_runner_v1.lua",
        LUA_NAME,
    }
    STEAM_SCRIPT.mkdir(parents=True, exist_ok=True)
    removed = []
    for p in STEAM_SCRIPT.glob("*.lua"):
        if p.name.startswith("toribash_") and p.name not in keep:
            p.unlink()
            removed.append(p.name)
    if PROJECT_LUA.exists():
        shutil.copy2(PROJECT_LUA, STEAM_LUA)
    print("kept:", sorted(keep))
    print("removed:", removed)


def write_candidate(generation: int, agent_idx: int) -> Path:
    ensure_dirs()
    parent = load_best_or_seed()
    child = parent if agent_idx == 0 else mutate_agent(parent, generation, agent_idx)
    child["generation"] = generation
    child["agent_id"] = agent_idx
    out = POP / f"walk_gru_live_v1_g{generation:04d}_a{agent_idx:02d}.json"
    out.write_text(json.dumps(child, indent=2), encoding="utf-8")
    export_current_agent(child, generation, agent_idx)
    print("candidate:", out)
    print("mutations:", child.get("mutation_count", 0))
    return out


def set_clipboard(text: str) -> bool:
    for cmd in (
        ["xclip", "-selection", "clipboard"],
        ["wl-copy"],
        ["xsel", "--clipboard", "--input"],
    ):
        try:
            p = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            if p.returncode == 0:
                return True
        except Exception:
            pass
    return False


def send_chat_command(cmd: str) -> None:
    # Reliable flow observed in earlier branches: t, Ctrl+A, Backspace, paste command, Enter.
    time.sleep(0.15)
    subprocess.run(["xdotool", "key", "t"], check=False)
    time.sleep(0.06)
    subprocess.run(["xdotool", "key", "ctrl+a"], check=False)
    time.sleep(0.03)
    subprocess.run(["xdotool", "key", "BackSpace"], check=False)
    time.sleep(0.05)
    if set_clipboard(cmd):
        subprocess.run(["xdotool", "key", "ctrl+v"], check=False)
    else:
        subprocess.run(["xdotool", "type", "--clearmodifiers", cmd], check=False)
    time.sleep(0.05)
    subprocess.run(["xdotool", "key", "Return"], check=False)


def reset_and_load_lua() -> None:
    send_chat_command("/reset")
    time.sleep(0.35)
    send_chat_command(f"/ls {LUA_NAME}")


def wait_result(timeout: float = RUN_TIMEOUT) -> Optional[dict]:
    start = time.time()
    while time.time() - start < timeout:
        if RESULT_JSON.exists() and RESULT_JSON.stat().st_size > 0:
            try:
                return json.loads(RESULT_JSON.read_text(encoding="utf-8"))
            except Exception:
                pass
        time.sleep(0.25)
    return None


def promote(agent_path: Path, score: float) -> None:
    BEST.mkdir(parents=True, exist_ok=True)
    dst = BEST / "walk_gru_live_v1_champion.json"
    agent = json.loads(agent_path.read_text(encoding="utf-8"))
    agent["parent_score"] = score
    dst.write_text(json.dumps(agent, indent=2), encoding="utf-8")
    state = load_state()
    state["best_agent"] = str(dst)
    state["best_score"] = score
    state.setdefault("history", []).append({"time": time.time(), "score": score, "agent": str(agent_path)})
    state["generation"] = int(state.get("generation", 1)) + 1
    save_state(state)
    print("PROMOTED", agent_path.name, "score", score)


def run_generation() -> None:
    setup()
    state = load_state()
    generation = int(state.get("generation", 1))
    print("Running generation", generation)
    best_score = None
    best_agent_path = None

    for i in range(POPULATION):
        path = write_candidate(generation, i)
        if RESULT_JSON.exists():
            RESULT_JSON.unlink()
        print(f"[{i+1}/{POPULATION}] load in Toribash...")
        reset_and_load_lua()
        result = wait_result()
        if result is None:
            print("  TIMEOUT no result")
            score = -999999.0
        else:
            score = float(result.get("score", -999999.0))
            print("  score", score, "reason", result.get("reason"), "dist", result.get("distance_y_body"), "frames", result.get("frames_alive"))
            result_out = path.with_suffix(".result.json")
            result_out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        if best_score is None or score > best_score:
            best_score = score
            best_agent_path = path

    if best_agent_path is not None and best_score is not None:
        old_best = state.get("best_score")
        if old_best is None or best_score >= float(old_best):
            promote(best_agent_path, best_score)
        else:
            print("Generation best did not beat champion:", best_score, "<", old_best)


def status() -> None:
    print(json.dumps(load_state(), indent=2))
    print("Lua:", STEAM_LUA, STEAM_LUA.exists())
    print("Agent:", AGENT_LUA, AGENT_LUA.exists())
    print("Result:", RESULT_JSON, RESULT_JSON.exists())


def usage() -> None:
    print("Usage:")
    print("  setup       copy Lua + create seed agent")
    print("  clean-lua   remove old ToribashAI Lua except upright/recovery/live walk")
    print("  candidate N export one candidate agent")
    print("  run         run one population with xdotool reset/load/scoring")
    print("  status      show state")


def main() -> None:
    random.seed(RANDOM_SEED + int(time.time() // 10))
    ensure_dirs()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "status"
    if cmd == "setup":
        setup()
    elif cmd == "clean-lua":
        clean_lua()
    elif cmd == "candidate":
        setup()
        idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        gen = int(load_state().get("generation", 1))
        write_candidate(gen, idx)
    elif cmd == "run":
        run_generation()
    elif cmd == "status":
        status()
    else:
        usage()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
