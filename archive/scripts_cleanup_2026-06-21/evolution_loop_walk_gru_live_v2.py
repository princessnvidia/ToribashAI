#!/usr/bin/env python3
"""
evolution_loop_walk_gru_live_v2.py

Live evolution from the working GRU/template-safe len265 walking seed.
V2 fixes the agent table format and uses the older reliable Lua workflow:
- Lua file: toribash_walk_gru_live_runner_v2.lua
- Agent file: toribash_walk_gru_live_agent_v2.lua
- Agent is exported both as globals and as a returned table.
- Chat automation uses "ls ..." (without slash) by default, because some Toribash builds report /ls as unknown.

Usage:
  cp ~/Téléchargements/*walk_gru_live_v2* ~/Documents/ToribashAI/scripts/
  cd ~/Documents/ToribashAI
  python3 scripts/evolution_loop_walk_gru_live_v2.py setup
  # in Toribash, if needed: ls toribash_walk_gru_live_runner_v2.lua
  python3 scripts/evolution_loop_walk_gru_live_v2.py run
"""
from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path.home() / "Documents/ToribashAI"
SCRIPTS = ROOT / "scripts"
GEN = ROOT / "generated_replays"
EVOL = ROOT / "evolution" / "walk_gru_live_v2"
POP = EVOL / "population"
BEST = EVOL / "best"
STATE = EVOL / "state.json"

STEAM = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
STEAM_SCRIPT = STEAM / "data" / "script"

LUA_NAME = "toribash_walk_gru_live_runner_v2.lua"
PROJECT_LUA = SCRIPTS / LUA_NAME
STEAM_LUA = STEAM_SCRIPT / LUA_NAME

AGENT_LUA = STEAM_SCRIPT / "toribash_walk_gru_live_agent_v2.lua"
AGENT_JSON = STEAM_SCRIPT / "toribash_walk_gru_live_agent_v2.json"
META_TXT = STEAM_SCRIPT / "toribash_walk_gru_live_meta_v2.txt"
RESULT_JSON = STEAM_SCRIPT / "toribash_walk_gru_live_result_v2.json"

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
RUN_TIMEOUT = 38.0
RANDOM_SEED = 552002

MUTATION_SCHEDULE = [
    (316, 500, 0.003, 0.002, 0.001),
    (501, 800, 0.006, 0.004, 0.002),
    (801, 99999, 0.009, 0.006, 0.003),
]
BALANCE_JOINTS = [1, 2, 3, 4, 7, 12, 13, 14, 15, 16, 17, 18, 19]
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
    d: Dict[int, int] = {}
    for i in range(0, len(nums) - 1, 2):
        j, v = nums[i], nums[i + 1]
        if 0 <= j < 20 and 1 <= v <= 4:
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
    commands = parse_rpl_actions(seed)
    return {
        "name": "walk_gru_live_seed_v2",
        "version": "walk_gru_live_v2",
        "source_rpl": str(seed),
        "generation": 0,
        "agent_id": "seed",
        "population": POPULATION,
        "protect_until": PROTECT_UNTIL,
        "max_frame": MAX_FRAME,
        "commands": commands,
        "parent_score": None,
    }


def load_best_or_seed() -> Agent:
    state = load_state()
    best_path = state.get("best_agent")
    if best_path and Path(best_path).exists():
        return json.loads(Path(best_path).read_text(encoding="utf-8"))
    seed_path = BEST / "walk_gru_live_seed_v2.json"
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
    child["name"] = f"walk_gru_live_v2_g{generation:04d}_a{agent_idx:02d}"
    child["version"] = "walk_gru_live_v2"
    child["generation"] = generation
    child["agent_id"] = agent_idx
    child["population"] = POPULATION
    child["protect_until"] = PROTECT_UNTIL
    child["max_frame"] = MAX_FRAME

    mutations = 0
    for cmd in child.get("commands", []):
        frame = int(cmd.get("frame", 0))
        mutate_rate, add_rate, drop_rate = rate_for_frame(frame)
        if mutate_rate <= 0:
            continue
        d = {int(j): int(v) for j, v in cmd.get("pairs", [])}
        for j in list(d.keys()):
            if random.random() < drop_rate:
                del d[j]
                mutations += 1
        for j in list(d.keys()):
            if random.random() < mutate_rate:
                old = d[j]
                new = max(1, min(4, old + random.choice([-1, 1])))
                if new != old:
                    d[j] = new
                    mutations += 1
        if random.random() < add_rate:
            j = random.choice(BALANCE_JOINTS)
            if j not in d:
                d[j] = random.randint(1, 4)
                mutations += 1
        cmd["pairs"] = [[j, d[j]] for j in sorted(d)]
    child["mutation_count"] = mutations
    return child


def lua_quote(s: object) -> str:
    return json.dumps(str(s), ensure_ascii=False)


def export_agent_lua(agent: Agent, path: Path) -> None:
    lines: List[str] = []
    lines.append("TORIBASHAI_WALK_GRU_AGENT = {")
    lines.append(f"  name = {lua_quote(agent.get('name', 'walk_gru_live'))},")
    lines.append(f"  version = {lua_quote(agent.get('version', 'walk_gru_live_v2'))},")
    lines.append(f"  gen = {int(agent.get('generation', 0))},")
    aid = agent.get("agent_id", "?")
    if isinstance(aid, int):
        lines.append(f"  agent_id = {aid},")
    else:
        lines.append(f"  agent_id = {lua_quote(aid)},")
    lines.append(f"  population = {int(agent.get('population', POPULATION))},")
    lines.append(f"  max_frame = {int(agent.get('max_frame', MAX_FRAME))},")
    lines.append(f"  protect_until = {int(agent.get('protect_until', PROTECT_UNTIL))},")
    lines.append("  actions = {")
    for cmd in sorted(agent.get("commands", []), key=lambda c: int(c.get("frame", 0))):
        pair_txt = ", ".join("{%d,%d}" % (int(j), int(v)) for j, v in cmd.get("pairs", []))
        lines.append(f"    {{ frame = {int(cmd.get('frame', 0))}, pairs = {{ {pair_txt} }} }},")
    lines.append("  }")
    lines.append("}")
    lines.append("TORIBASHAI_AGENT = TORIBASHAI_WALK_GRU_AGENT")
    lines.append("return TORIBASHAI_WALK_GRU_AGENT")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_current_agent(agent: Agent, generation: int, agent_idx: int) -> None:
    export_agent_lua(agent, AGENT_LUA)
    AGENT_JSON.write_text(json.dumps(agent, indent=2), encoding="utf-8")
    META_TXT.write_text(
        f"gen={generation}\nagent={agent_idx}\npopulation={POPULATION}\nparent_score={agent.get('parent_score')}\nname={agent.get('name')}\n",
        encoding="utf-8",
    )
    if RESULT_JSON.exists():
        RESULT_JSON.unlink()


def setup() -> None:
    ensure_dirs()
    if not PROJECT_LUA.exists():
        raise FileNotFoundError(f"Missing Lua in scripts: {PROJECT_LUA}")
    shutil.copy2(PROJECT_LUA, STEAM_LUA)
    seed = make_seed_agent()
    seed_path = BEST / "walk_gru_live_seed_v2.json"
    seed_path.write_text(json.dumps(seed, indent=2), encoding="utf-8")
    state = load_state()
    state["best_agent"] = str(seed_path)
    state["best_score"] = None
    state["generation"] = max(1, int(state.get("generation", 1)))
    save_state(state)
    export_current_agent(seed, 0, 0)
    print("Lua copied:", STEAM_LUA)
    print("Seed actions:", len(seed["commands"]))
    print("Seed source:", seed.get("source_rpl"))
    print("Agent exported:", AGENT_LUA)


def set_clipboard(text: str) -> bool:
    for cmd in (["xclip", "-selection", "clipboard"], ["wl-copy"], ["xsel", "--clipboard", "--input"]):
        try:
            p = subprocess.run(cmd, input=text.encode(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            if p.returncode == 0:
                return True
        except Exception:
            pass
    return False


def send_chat_command(cmd: str) -> None:
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
    # Old reliable flow but without slash for ls.
    send_chat_command("/reset")
    time.sleep(0.45)
    send_chat_command(f"ls {LUA_NAME}")


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


def write_candidate(generation: int, agent_idx: int) -> Path:
    parent = load_best_or_seed()
    child = parent if agent_idx == 0 else mutate_agent(parent, generation, agent_idx)
    child["generation"] = generation
    child["agent_id"] = agent_idx
    out = POP / f"walk_gru_live_v2_g{generation:04d}_a{agent_idx:02d}.json"
    out.write_text(json.dumps(child, indent=2), encoding="utf-8")
    export_current_agent(child, generation, agent_idx)
    print("candidate:", out)
    print("mutations:", child.get("mutation_count", 0))
    return out


def promote(agent_path: Path, result: dict) -> None:
    state = load_state()
    score = float(result.get("score", -1e9))
    best_score = state.get("best_score")
    if best_score is None or score > float(best_score):
        dst = BEST / "walk_gru_live_v2_champion.json"
        shutil.copy2(agent_path, dst)
        data = json.loads(dst.read_text(encoding="utf-8"))
        data["parent_score"] = score
        dst.write_text(json.dumps(data, indent=2), encoding="utf-8")
        state["best_score"] = score
        state["best_agent"] = str(dst)
        print("NEW BEST", score, dst)
    state.setdefault("history", []).append(result)
    save_state(state)


def run_generation() -> None:
    ensure_dirs()
    if PROJECT_LUA.exists():
        shutil.copy2(PROJECT_LUA, STEAM_LUA)
    state = load_state()
    generation = int(state.get("generation", 1))
    print("Generation", generation)
    for idx in range(POPULATION):
        agent_path = write_candidate(generation, idx)
        reset_and_load_lua()
        result = wait_result()
        if result is None:
            print("TIMEOUT", idx)
            result = {"score": -999999, "reason": "timeout", "gen": generation, "agent": idx}
        print("RESULT", idx, json.dumps(result, ensure_ascii=False))
        promote(agent_path, result)
        time.sleep(0.3)
    state = load_state()
    state["generation"] = generation + 1
    save_state(state)
    print("Next generation:", generation + 1)


def clean_lua() -> None:
    keep = {"toribash_upright_runner_v18.lua", "toribash_recovery_runner_v1.lua", LUA_NAME}
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


def main() -> None:
    random.seed(RANDOM_SEED)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "help"
    if cmd == "setup":
        setup()
    elif cmd == "clean-lua":
        clean_lua()
    elif cmd == "candidate":
        gen = int(sys.argv[2]) if len(sys.argv) > 2 else load_state().get("generation", 1)
        idx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
        write_candidate(gen, idx)
    elif cmd == "run":
        while True:
            run_generation()
    elif cmd == "one-gen":
        run_generation()
    else:
        print("Usage: setup | clean-lua | candidate [gen idx] | one-gen | run")


if __name__ == "__main__":
    main()
