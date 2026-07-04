#!/usr/bin/env python3
"""
evolution_loop_xioi_gru_template_v55.py

ToribashAI V55: RPL-template-safe evolution around the GRU/len265 walking champion.

Important idea:
- Lua DOES NOT control Tori joints. It only scores the replay.
- Python generates full .rpl candidates from a working champion template.
- Frames 0..315 are protected: this is the launch / getting into the walk.
- Mutations are tiny and only after the launch, preserving the GRU/template-safe gait.

Manual workflow:
  python3 scripts/evolution_loop_xioi_gru_template_v55.py generate
  # open Toribash, load scorer once: /ls toribash_xioi_gru_scorer_v55.lua
  # inspect candidates or run them, then promote:
  python3 scripts/evolution_loop_xioi_gru_template_v55.py promote xioi_gru_v55_g001_c07.rpl

Semi-auto workflow, if your Toribash replay UI is focused enough for xdotool automation:
  python3 scripts/evolution_loop_xioi_gru_template_v55.py run
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
GEN_DIR = ROOT / "generated_replays"
EVOL_DIR = ROOT / "evolution" / "xioi_gru_template_v55"
POP_DIR = EVOL_DIR / "population"
BEST_DIR = EVOL_DIR / "best"
STATE_PATH = EVOL_DIR / "state.json"
SCRIPTS_DIR = ROOT / "scripts"

STEAM_TORIBASH = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
STEAM_REPLAY = STEAM_TORIBASH / "replay"
STEAM_PARKOUR = STEAM_REPLAY / "parkour"
STEAM_SCRIPT = STEAM_TORIBASH / "data" / "script"
RESULT_PATH = STEAM_SCRIPT / "toribash_xioi_gru_score_v55.json"
META_PATH = STEAM_SCRIPT / "toribash_xioi_gru_current_v55.txt"
LUA_NAME = "toribash_xioi_gru_scorer_v55.lua"
PROJECT_LUA = SCRIPTS_DIR / LUA_NAME

# Best known walking templates, in preference order.
TEMPLATE_CANDIDATES = [
    GEN_DIR / "xioi_loop_len265_gru_v54_seed048.rpl",
    GEN_DIR / "xioi_loop_len265_gru_v54_seed008.rpl",
    GEN_DIR / "xioi_loop_len265_gru_v53_free_template_safe.rpl",
    GEN_DIR / "xioi_loop_len265_gru_v53_teacher_template_safe.rpl",
    GEN_DIR / "xioi_loop_len265_champion_v51.rpl",
    GEN_DIR / "xioi_loop_phase_v50_len265.rpl",
]

PROTECT_UNTIL_FRAME = 315
POPULATION = 10
GENERATIONS_PER_RUN = 12
RANDOM_SEED = 552655

# Tiny mutation schedule. We do not touch the launch. We barely touch the walk.
MUTATION_RATES = [
    # (frame_min, frame_max, per_pair_rate, add_rate, drop_rate)
    (316, 500, 0.006, 0.004, 0.002),
    (501, 900, 0.012, 0.008, 0.004),
    (901, 99999, 0.018, 0.012, 0.006),
]

JOINTS_BALANCE = [1, 2, 3, 4, 7, 12, 13, 14, 15, 16, 17, 18, 19]
JOINTS_ALL = list(range(20))

FrameBlock = List[str]
JointPairs = List[Tuple[int, int]]

FRAME_RE = re.compile(r"^FRAME\s+(-?\d+)\s*;")
FIGHTNAME_RE = re.compile(r"^FIGHTNAME\s+0\s*;")
JOINT0_RE = re.compile(r"^JOINT\s+0\s*;\s*(.*)$")


def ensure_dirs() -> None:
    for p in [GEN_DIR, EVOL_DIR, POP_DIR, BEST_DIR, STEAM_REPLAY, STEAM_PARKOUR, STEAM_SCRIPT]:
        p.mkdir(parents=True, exist_ok=True)


def find_template() -> Path:
    for p in TEMPLATE_CANDIDATES:
        if p.exists():
            return p
    found = sorted(GEN_DIR.glob("*v54*.rpl")) + sorted(GEN_DIR.glob("*v53*.rpl")) + sorted(GEN_DIR.glob("*len265*.rpl"))
    if found:
        return found[0]
    raise FileNotFoundError("No V53/V54/len265 walking template found in generated_replays.")


def read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_pairs(payload: str) -> JointPairs:
    nums = [int(x) for x in re.findall(r"-?\d+", payload)]
    pairs: JointPairs = []
    for i in range(0, len(nums) - 1, 2):
        j, v = nums[i], nums[i + 1]
        if 0 <= j <= 19 and 1 <= v <= 4:
            pairs.append((j, v))
    return pairs


def pairs_to_line(pairs: JointPairs) -> str:
    if not pairs:
        return "JOINT 0;"
    flat: List[str] = []
    seen = set()
    for j, v in pairs:
        if j in seen:
            continue
        seen.add(j)
        flat.extend([str(j), str(v)])
    return "JOINT 0; " + " ".join(flat)


def current_frame(line: str) -> Optional[int]:
    m = FRAME_RE.match(line)
    return int(m.group(1)) if m else None


def mutation_schedule(frame: int) -> Tuple[float, float, float]:
    for lo, hi, rate, add, drop in MUTATION_RATES:
        if lo <= frame <= hi:
            return rate, add, drop
    return 0.0, 0.0, 0.0


def mutate_pairs(pairs: JointPairs, frame: int, rng: random.Random) -> JointPairs:
    if frame <= PROTECT_UNTIL_FRAME:
        return pairs[:]

    pair_rate, add_rate, drop_rate = mutation_schedule(frame)
    if pair_rate <= 0:
        return pairs[:]

    out: Dict[int, int] = {j: v for j, v in pairs}

    # Prefer balance joints after the launch; keep leg rhythm mostly intact.
    for j in list(out.keys()):
        if rng.random() < drop_rate and j in JOINTS_BALANCE:
            del out[j]
            continue
        if rng.random() < pair_rate:
            old = out[j]
            # tiny adjacent change, not random teleport
            options = [old]
            if old > 1:
                options.append(old - 1)
            if old < 4:
                options.append(old + 1)
            out[j] = rng.choice(options)

    if rng.random() < add_rate:
        j = rng.choice(JOINTS_BALANCE)
        if j not in out:
            out[j] = rng.choice([1, 2, 3, 4])

    return sorted(out.items())


def mutate_rpl(template: Path, output: Path, fightname: str, rng: random.Random) -> Dict[str, int]:
    lines = read_lines(template)
    out: List[str] = []
    frame = 0
    mutations = 0
    joint_lines = 0

    for line in lines:
        fr = current_frame(line)
        if fr is not None:
            frame = fr
            out.append(line)
            continue

        if FIGHTNAME_RE.match(line):
            out.append(f"FIGHTNAME 0; {fightname}")
            continue

        m = JOINT0_RE.match(line)
        if m:
            joint_lines += 1
            old_pairs = parse_pairs(m.group(1))
            new_pairs = mutate_pairs(old_pairs, frame, rng)
            if new_pairs != old_pairs:
                mutations += 1
            out.append(pairs_to_line(new_pairs))
            continue

        out.append(line)

    write_lines(output, out)
    return {"mutated_joint_lines": mutations, "joint_lines": joint_lines}


def copy_to_steam(path: Path) -> None:
    for dest_dir in [STEAM_REPLAY, STEAM_PARKOUR]:
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest_dir / path.name)


def install_lua() -> None:
    if not PROJECT_LUA.exists():
        print(f"WARNING: missing Lua in scripts: {PROJECT_LUA}")
        return
    shutil.copy2(PROJECT_LUA, STEAM_SCRIPT / PROJECT_LUA.name)
    print("Lua copied:", STEAM_SCRIPT / PROJECT_LUA.name)


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    template = find_template()
    return {
        "version": 55,
        "generation": 0,
        "champion": str(template),
        "best_score": None,
        "history": [],
    }


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def generate_population() -> None:
    ensure_dirs()
    install_lua()
    state = load_state()
    gen = int(state.get("generation", 0)) + 1
    parent = Path(state["champion"])
    if not parent.exists():
        parent = find_template()

    rng = random.Random(RANDOM_SEED + gen)
    made = []

    # Parent candidate first
    parent_name = f"xioi_gru_v55_g{gen:03d}_c00_PARENT.rpl"
    parent_out = GEN_DIR / parent_name
    shutil.copy2(parent, parent_out)
    # Fix fightname on parent copy
    lines = [f"FIGHTNAME 0; {parent_out.stem}" if FIGHTNAME_RE.match(l) else l for l in read_lines(parent_out)]
    write_lines(parent_out, lines)
    copy_to_steam(parent_out)
    made.append({"file": parent_name, "parent": True, "mutated_joint_lines": 0})

    for i in range(1, POPULATION + 1):
        name = f"xioi_gru_v55_g{gen:03d}_c{i:02d}.rpl"
        out = GEN_DIR / name
        stats = mutate_rpl(parent, out, Path(name).stem, random.Random(rng.randint(0, 10**9)))
        copy_to_steam(out)
        made.append({"file": name, "parent": False, **stats})

    state["generation"] = gen
    state["last_population"] = made
    state["population_size"] = POPULATION
    save_state(state)

    print(f"Generated V55 generation {gen:03d} from parent:", parent)
    for m in made:
        print(" ", m)
    print("\nIn Toribash load scorer once:")
    print(f"  /ls {LUA_NAME}")
    print("Then inspect candidates in replay/parkour or use promote after visual selection.")


def promote(candidate_name: str, score: Optional[float] = None) -> None:
    ensure_dirs()
    state = load_state()
    src = GEN_DIR / candidate_name
    if not src.exists():
        # Maybe user passed only c07 and generation is known
        matches = sorted(GEN_DIR.glob(f"*{candidate_name}*.rpl"))
        if len(matches) == 1:
            src = matches[0]
        else:
            raise FileNotFoundError(f"Cannot find candidate: {candidate_name}")
    champ = BEST_DIR / "xioi_gru_template_v55_champion.rpl"
    shutil.copy2(src, champ)
    copy_to_steam(champ)
    state["champion"] = str(champ)
    state["best_score"] = score if score is not None else state.get("best_score")
    state.setdefault("history", []).append({"generation": state.get("generation"), "champion": src.name, "score": score})
    save_state(state)
    print("Promoted:", src)
    print("Champion:", champ)


def write_meta(gen: int, agent: str, candidate: str) -> None:
    META_PATH.write_text(f"gen={gen}\nagent={agent}\ncandidate={candidate}\n", encoding="utf-8")


def load_result(timeout: float = 0.0) -> Optional[dict]:
    if timeout:
        end = time.time() + timeout
        while time.time() < end:
            r = load_result(0.0)
            if r and r.get("score") is not None:
                return r
            time.sleep(0.25)
    if not RESULT_PATH.exists():
        return None
    try:
        return json.loads(RESULT_PATH.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return None


def xdotool_available() -> bool:
    return shutil.which("xdotool") is not None


def run_cmd(cmd: List[str], check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def play_replay_best_effort(replay_name: str) -> None:
    """Best-effort UI automation. Toribash CLI replay loading is inconsistent; this uses chat command."""
    if not xdotool_available():
        print("xdotool not found; open replay manually:", replay_name)
        return
    # Focus assumed on Toribash. Use robust chat paste pattern from earlier project.
    cmd = f"/lp {replay_name}"
    try:
        run_cmd(["xdotool", "key", "t"])
        time.sleep(0.08)
        run_cmd(["xdotool", "key", "ctrl+a"])
        run_cmd(["xdotool", "key", "BackSpace"])
        run_cmd(["xdotool", "type", "--clearmodifiers", cmd])
        run_cmd(["xdotool", "key", "Return"])
        time.sleep(0.2)
        # Start/space quickly after load
        run_cmd(["xdotool", "key", "space"])
    except Exception as e:
        print("xdotool automation failed:", e)


def run_generation() -> None:
    """Semi-automatic scorer loop. May need user focus on Toribash window."""
    generate_population()
    state = load_state()
    gen = int(state.get("generation", 0))
    pop = state.get("last_population", [])
    results = []
    print("Starting semi-auto scoring. Keep Toribash focused with scorer loaded.")
    print(f"If needed in Toribash: /ls {LUA_NAME}")
    time.sleep(2)
    for idx, cand in enumerate(pop):
        name = cand["file"]
        RESULT_PATH.write_text('{"status":"pending"}\n', encoding="utf-8")
        write_meta(gen, f"{idx:02d}/{len(pop)-1:02d}", name)
        print("Candidate", idx, name)
        play_replay_best_effort(name)
        result = load_result(timeout=20.0)
        if not result or result.get("score") is None:
            print("  no score; mark very low")
            score = -999999.0
            result = {"score": score, "reason": "no_score"}
        else:
            score = float(result.get("score", -999999.0))
            print("  score", score, "reason", result.get("reason"))
        results.append({"file": name, "score": score, "result": result})

    results.sort(key=lambda x: x["score"], reverse=True)
    summary_path = EVOL_DIR / f"generation_{gen:03d}_summary.json"
    summary_path.write_text(json.dumps({"generation": gen, "results": results}, indent=2), encoding="utf-8")
    print("Best:", results[0])
    promote(results[0]["file"], results[0]["score"])


def clean_lua_keep() -> None:
    keep = {"toribash_upright_runner_v18.lua", "toribash_recovery_runner_v1.lua", LUA_NAME}
    for p in STEAM_SCRIPT.glob("*.lua"):
        if p.name.startswith("toribash") and p.name not in keep:
            try:
                p.unlink()
                print("removed", p.name)
            except Exception as e:
                print("could not remove", p, e)
    install_lua()


def status() -> None:
    state = load_state()
    print(json.dumps(state, indent=2))
    print("Template:", find_template())
    print("Lua:", STEAM_SCRIPT / LUA_NAME, "exists=", (STEAM_SCRIPT / LUA_NAME).exists())


def main() -> None:
    ensure_dirs()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "generate"
    if cmd == "generate":
        generate_population()
    elif cmd == "promote":
        if len(sys.argv) < 3:
            raise SystemExit("Usage: promote <candidate.rpl> [score]")
        score = float(sys.argv[3]) if len(sys.argv) >= 4 else None
        promote(sys.argv[2], score)
    elif cmd == "run":
        run_generation()
    elif cmd == "clean-lua":
        clean_lua_keep()
    elif cmd == "status":
        status()
    else:
        raise SystemExit("Commands: generate | promote | run | clean-lua | status")


if __name__ == "__main__":
    main()
