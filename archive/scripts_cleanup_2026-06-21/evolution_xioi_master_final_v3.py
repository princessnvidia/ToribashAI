#!/usr/bin/env python3
"""
evolution_xioi_master_final_v3.py

Xioi Master Final V3
- Nettoie les anciens Lua Steam sauf toribash_upright_runner_v18.lua et toribash_recovery_runner_v1.lua.
- Part du champion V30 promu / xioi_v30_23_mut.rpl.
- Population 10 par génération.
- Protège les frames < 70.
- Mutations surtout après frame 70, avec biais épaules/bassin/bras/jambes opposés.
- Score Lua sur distance Y du buste / zone au-dessus des genoux.

Usage:
    cd ~/Documents/ToribashAI
    python3 scripts/evolution_xioi_master_final_v3.py
"""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
SCRIPTS = ROOT / "scripts"
OUT_DIR = ROOT / "generated_replays" / "xioi_master_final_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TORIBASH_SCRIPT_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"
)
TORIBASH_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)

PROJECT_LUA = SCRIPTS / "toribash_xioi_master_final_v3.lua"
STEAM_LUA = TORIBASH_SCRIPT_DIR / "toribash_xioi_master_final_v3.lua"
STEAM_AGENT_LUA = TORIBASH_SCRIPT_DIR / "xioi_master_final_v3_agent_current.lua"
STEAM_RESULT = TORIBASH_SCRIPT_DIR / "xioi_master_final_v3_result.json"

CHAMPION_AGENT = OUT_DIR / "xioi_master_final_v3_champion_agent.json"
CHAMPION_RPL = OUT_DIR / "xioi_master_final_v3_champion.rpl"
HISTORY = OUT_DIR / "xioi_master_final_v3_history.jsonl"

# Parent préféré : V3 champion si relance, sinon le V30_23 sain qui fait encore 3 pas.
PARENT_CANDIDATES = [
    OUT_DIR / "xioi_master_final_v3_champion.rpl",
    ROOT / "generated_replays" / "xioi_v30_23_mut.rpl",
    ROOT / "generated_replays" / "xioi_v30_champion.rpl",
    ROOT / "generated_replays" / "xioi_source_template_v28.rpl",
]

POPULATION = 10
GENERATIONS = 40
RESULT_TIMEOUT_SEC = 8.5
RESET_WAIT = 0.18
LOAD_WAIT = 0.16
TURNFRAMES = 5
MAX_TICKS = 340
CORRECT_FROM_FRAME = 70

# V3: on préserve exactement le démarrage qui marche déjà.
# 0-70 verrouillé. 70-140 micro-corrections. 140+ mutations légères.
PROTECTED_BEFORE = 70
MICRO_END = 140
EARLY_FOCUS_END = 220
MUTATE_RATE_MICRO = 0.006
MUTATE_RATE_MAIN = 0.032
MUTATE_RATE_LATE = 0.022
ADD_PAIR_RATE = 0.010
DROP_PAIR_RATE = 0.006
COUNTERSWING_RATE = 0.12

# Groupes approximatifs Toribash, suffisants pour créer des oppositions bras/jambes.
ARMS_SHOULDERS = [4, 5, 6, 7, 8, 9]
CORE_HIPS = [2, 3, 14, 15]
LEGS = [14, 15, 16, 17, 18, 19]
BALANCE_JOINTS = sorted(set(ARMS_SHOULDERS + CORE_HIPS + LEGS))
MICRO_BALANCE_JOINTS = sorted(set(ARMS_SHOULDERS + [2, 3, 14, 15]))
ALL_JOINTS = list(range(20))
VALUES = [1, 2, 3, 4]

LUA_COMMAND = "/ls toribash_xioi_master_final_v3.lua"
RESET_COMMAND = "/reset"


@dataclass
class Action:
    frame: int
    pairs: list[list[int]]


def clean_steam_lua() -> None:
    """Remove old ToribashAI lua scripts from Steam, preserving the two requested runners."""
    TORIBASH_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    keep = {
        "toribash_upright_runner_v18.lua",
        "toribash_recovery_runner_v1.lua",
        "toribash_xioi_master_final_v3.lua",
        "xioi_master_final_v3_agent_current.lua",
        "toribash_upright_runner_v18.lua",
        "toribash_recovery_runner_v1.lua",
    }
    removed = []
    for path in TORIBASH_SCRIPT_DIR.glob("*.lua"):
        if path.name in keep:
            continue
        # On cible les scripts de nos expérimentations, sans supprimer des scripts système éventuels hors ToribashAI.
        name = path.name.lower()
        if (
            name.startswith("toribash_")
            or "xioi" in name
            or "parkour" in name
            or "walking" in name
            or "gru" in name
        ):
            try:
                path.unlink()
                removed.append(path.name)
            except FileNotFoundError:
                pass
    if removed:
        print("Steam Lua removed:", ", ".join(sorted(removed)))
    else:
        print("Steam Lua cleanup: nothing removed")


def deploy_lua() -> None:
    if not PROJECT_LUA.exists():
        raise FileNotFoundError(PROJECT_LUA)
    TORIBASH_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_LUA, STEAM_LUA)
    print("Lua deployed:", STEAM_LUA)


def focus_toribash() -> None:
    subprocess.run(["xdotool", "search", "--name", "Toribash", "windowactivate", "--sync"], check=False)
    time.sleep(0.05)


def send_chat_command(command: str) -> None:
    focus_toribash()
    try:
        p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
        p.communicate(command.encode("utf-8"))
        subprocess.run(["xdotool", "key", "t"], check=False)
        time.sleep(0.035)
        subprocess.run(["xdotool", "key", "ctrl+a"], check=False)
        time.sleep(0.015)
        subprocess.run(["xdotool", "key", "BackSpace"], check=False)
        time.sleep(0.015)
        subprocess.run(["xdotool", "key", "ctrl+v"], check=False)
        time.sleep(0.015)
        subprocess.run(["xdotool", "key", "Return"], check=False)
    except Exception:
        subprocess.run(["xdotool", "key", "t"], check=False)
        subprocess.run(["xdotool", "type", "--delay", "1", command], check=False)
        subprocess.run(["xdotool", "key", "Return"], check=False)


def press_space() -> None:
    focus_toribash()
    subprocess.run(["xdotool", "key", "space"], check=False)
    time.sleep(0.025)

def find_parent_rpl() -> Path:
    for p in PARENT_CANDIDATES:
        if p.exists():
            return p
    matches = sorted((ROOT / "generated_replays").glob("*v30*23*mut*.rpl"))
    if matches:
        return matches[0]
    raise FileNotFoundError("Aucun parent V30/V38 trouvé. Vérifie xioi_v30_23_mut.rpl ou le champion promu.")


def parse_rpl_actions(path: Path) -> tuple[list[str], list[Action]]:
    header: list[str] = []
    actions_by_frame: dict[int, dict[int, int]] = {}
    current_frame: int | None = None

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    seen_frame = False
    for line in lines:
        if line.startswith("FRAME "):
            seen_frame = True
            m = re.match(r"FRAME\s+(-?\d+)", line)
            if m:
                current_frame = int(m.group(1))
                actions_by_frame.setdefault(current_frame, {})
            continue
        if not seen_frame:
            header.append(line)
        if current_frame is not None and line.startswith("JOINT 0;"):
            nums = [int(x) for x in re.findall(r"-?\d+", line.split(";", 1)[1])]
            # Format: JOINT 0; joint value [joint value...]
            for i in range(0, len(nums) - 1, 2):
                j, v = nums[i], nums[i + 1]
                if 0 <= j <= 19 and 1 <= v <= 4:
                    actions_by_frame[current_frame][j] = v

    actions = [Action(frame=fr, pairs=[[j, v] for j, v in sorted(pairs.items())]) for fr, pairs in sorted(actions_by_frame.items())]
    if not actions:
        raise RuntimeError(f"Aucune action JOINT trouvée dans {path}")
    return header, actions


def actions_to_dict(actions: list[Action]) -> list[dict[str, Any]]:
    return [{"frame": int(a.frame), "pairs": [[int(j), int(v)] for j, v in a.pairs]} for a in actions]


def normalize_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for a in actions:
        clean: dict[int, int] = {}
        for j, v in a.get("pairs", []):
            j, v = int(j), int(v)
            if 0 <= j <= 19 and 1 <= v <= 4:
                clean[j] = v
        out.append({"frame": int(a["frame"]), "pairs": [[j, clean[j]] for j in sorted(clean)]})
    out.sort(key=lambda x: x["frame"])
    return out


def lua_quote(s: str) -> str:
    return json.dumps(str(s), ensure_ascii=False)


def write_agent_lua(agent: dict[str, Any], path: Path) -> None:
    lines = ["return {"]
    for key in ["name", "run_id"]:
        lines.append(f"  {key} = {lua_quote(agent.get(key, ''))},")
    for key in ["generation", "candidate", "population", "generations", "turnframes", "max_ticks", "correct_from_frame"]:
        lines.append(f"  {key} = {int(agent.get(key, 0))},")
    lines.append("  enable_pilot = true,")
    lines.append("  actions = {")
    for a in agent["actions"]:
        pairs = ", ".join("{%d,%d}" % (int(j), int(v)) for j, v in a.get("pairs", []))
        lines.append(f"    {{ frame = {int(a['frame'])}, pairs = {{ {pairs} }} }},")
    lines.append("  }")
    lines.append("}\n")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_rpl_from_actions(header: list[str], actions: list[dict[str, Any]], out_path: Path, fightname: str) -> None:
    # Garde le contexte physique du parent, remplace juste les blocs FRAME/JOINT.
    clean_header = []
    for line in header:
        if line.startswith("FIGHTNAME"):
            clean_header.append(f"FIGHTNAME 0; {fightname}")
        else:
            clean_header.append(line)
    text = "\n".join(clean_header).rstrip() + "\n\n"
    for a in actions:
        text += f"FRAME {int(a['frame'])};\n"
        pairs = a.get("pairs", [])
        if pairs:
            flat = " ".join(f"{int(j)} {int(v)}" for j, v in pairs)
            text += f"JOINT 0; {flat}\n"
        text += "\n"
    out_path.write_text(text, encoding="utf-8")


def weighted_frame_choice(frames: list[int]) -> int:
    candidates = [f for f in frames if f >= PROTECTED_BEFORE]
    if not candidates:
        return random.choice(frames)
    # V3: focus frames 70-220, avec surtout micro-corrections 70-140.
    r = random.random()
    micro = [f for f in candidates if f <= MICRO_END]
    early = [f for f in candidates if MICRO_END < f <= EARLY_FOCUS_END]
    mid = [f for f in candidates if EARLY_FOCUS_END < f <= 280]
    late = [f for f in candidates if f > 280]
    if r < 0.55 and micro:
        return random.choice(micro)
    if r < 0.82 and early:
        return random.choice(early)
    if r < 0.95 and mid:
        return random.choice(mid)
    if late:
        return random.choice(late)
    return random.choice(candidates)


def opposite_value(v: int) -> int:
    return {1: 4, 2: 3, 3: 2, 4: 1}.get(int(v), random.choice(VALUES))


def clamp_value(v: int) -> int:
    return max(1, min(4, int(v)))


def mutate_agent(parent_actions: list[dict[str, Any]], generation: int, candidate: int) -> dict[str, Any]:
    actions = json.loads(json.dumps(parent_actions))
    by_frame = {int(a["frame"]): {int(j): int(v) for j, v in a.get("pairs", [])} for a in actions}
    frames = sorted(by_frame)

    strength = 1 + min(3, generation // 8)
    edits = random.randint(1, 3 + strength)

    for _ in range(edits):
        fr = weighted_frame_choice(frames)
        pairs = by_frame[fr]
        if fr <= MICRO_END:
            rate = MUTATE_RATE_MICRO
        elif fr <= EARLY_FOCUS_END:
            rate = MUTATE_RATE_MAIN
        else:
            rate = MUTATE_RATE_LATE

        # mutation locale surtout sur épaules/bassin.
        # V3: entre 70 et 140, on ne touche presque pas aux jambes: micro-équilibre seulement.
        if fr <= MICRO_END:
            target_pool = MICRO_BALANCE_JOINTS
            prefer_existing = 0.90
        else:
            target_pool = BALANCE_JOINTS if random.random() < 0.86 else ALL_JOINTS
            prefer_existing = 0.68

        if random.random() < prefer_existing and pairs:
            existing = [j for j in pairs if j in target_pool] or list(pairs)
            j = random.choice(existing)
        else:
            j = random.choice(target_pool)

        if random.random() < DROP_PAIR_RATE and j in pairs and fr >= 150:
            pairs.pop(j, None)
        else:
            old = pairs.get(j, random.choice(VALUES))
            if fr <= MICRO_END:
                # ultra-doux: 75% du temps on ne change rien, sinon juste +1/-1.
                if random.random() < 0.75:
                    new = old
                else:
                    new = clamp_value(old + random.choice([-1, 1]))
            elif random.random() < rate * 2.2:
                new = random.choice([v for v in VALUES if v != old])
            else:
                new = clamp_value(old + random.choice([-1, 1]))
            pairs[j] = new

        # Ajouter parfois un couple bras/jambes opposés pour épaules/bassin, jamais trop tôt.
        if random.random() < ADD_PAIR_RATE and fr >= 145:
            pairs[random.choice(target_pool)] = random.choice(VALUES)

    # Contre-swing structuré après frame 150 : bras et jambes opposés, pour ne pas casser les premiers pas.
    if random.random() < COUNTERSWING_RATE:
        late_frames = [f for f in frames if f >= 150]
        fr = random.choice(late_frames) if late_frames else weighted_frame_choice(frames)
        pairs = by_frame[fr]
        leg = random.choice([14, 15, 16, 17])
        arm = random.choice([4, 5, 6, 7, 8, 9])
        leg_v = pairs.get(leg, random.choice([2, 3]))
        pairs[arm] = opposite_value(leg_v)
        pairs[leg] = leg_v
        # bassin doux dans le sens inverse du bras
        pairs[random.choice([2, 3])] = random.choice([2, 3])

    new_actions = [{"frame": fr, "pairs": [[j, pairs[j]] for j in sorted(pairs)]} for fr, pairs in sorted(by_frame.items())]
    return {
        "name": "xioi_master_final_v3",
        "run_id": f"g{generation:04d}_c{candidate:02d}_{random.randint(0, 999999):06d}",
        "generation": generation,
        "candidate": candidate,
        "population": POPULATION,
        "generations": GENERATIONS,
        "turnframes": TURNFRAMES,
        "max_ticks": MAX_TICKS,
        "correct_from_frame": CORRECT_FROM_FRAME,
        "actions": normalize_actions(new_actions),
    }


def read_result(expected_run_id: str, timeout: float = RESULT_TIMEOUT_SEC) -> dict[str, Any] | None:
    start = time.time()
    while time.time() - start < timeout:
        if STEAM_RESULT.exists() and STEAM_RESULT.stat().st_size > 0:
            try:
                data = json.loads(STEAM_RESULT.read_text(encoding="utf-8"))
                if str(data.get("run_id")) == expected_run_id:
                    return data
            except Exception:
                pass
        time.sleep(0.08)
    return None


def evaluate(agent: dict[str, Any], reload_lua: bool = False) -> dict[str, Any]:
    if STEAM_RESULT.exists():
        STEAM_RESULT.unlink()
    write_agent_lua(agent, STEAM_AGENT_LUA)

    send_chat_command(RESET_COMMAND)
    time.sleep(RESET_WAIT)
    if reload_lua:
        send_chat_command(LUA_COMMAND)
        time.sleep(LOAD_WAIT)
    else:
        # Le Lua recharge sur new_game, mais /ls garantit le bon fichier si Toribash a gardé un ancien hook.
        send_chat_command(LUA_COMMAND)
        time.sleep(LOAD_WAIT)

    # V3: auto-start plus rapide juste après reset/load.
    press_space()

    result = read_result(agent["run_id"])
    if result is None:
        return {
            "score": -999999.0,
            "timeout": True,
            "run_id": agent["run_id"],
            "generation": agent["generation"],
            "candidate": agent["candidate"],
        }
    result["timeout"] = False
    return result


def save_history(result: dict[str, Any]) -> None:
    with HISTORY.open("a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> None:
    random.seed()
    parent_rpl = find_parent_rpl()
    print("Parent:", parent_rpl)

    header, parent_actions_raw = parse_rpl_actions(parent_rpl)
    parent_actions = actions_to_dict(parent_actions_raw)
    print("Parent actions:", len(parent_actions), "frames", parent_actions[0]["frame"], "->", parent_actions[-1]["frame"])

    clean_steam_lua()
    deploy_lua()

    input("Entrée quand Toribash est ouvert, puis laisse tourner l'évolution... ")

    # Si un champion existe déjà, on le reprend.
    if CHAMPION_AGENT.exists():
        champ = json.loads(CHAMPION_AGENT.read_text(encoding="utf-8"))
        parent_actions = normalize_actions(champ["actions"])
        print("Continuing from previous champion:", CHAMPION_AGENT)

    parent_agent = {
        "name": "xioi_master_final_v3",
        "run_id": f"parent_{random.randint(0, 999999):06d}",
        "generation": 0,
        "candidate": 0,
        "population": POPULATION,
        "generations": GENERATIONS,
        "turnframes": TURNFRAMES,
        "max_ticks": MAX_TICKS,
        "correct_from_frame": CORRECT_FROM_FRAME,
        "actions": parent_actions,
    }
    best_agent = parent_agent
    best_result = evaluate(parent_agent, reload_lua=True)
    print("Initial:", best_result)
    save_history(best_result)

    best_score = float(best_result.get("score", -999999.0))
    write_rpl_from_actions(header, best_agent["actions"], CHAMPION_RPL, "xioi_master_final_v3_champion")
    CHAMPION_AGENT.write_text(json.dumps(best_agent, indent=2), encoding="utf-8")

    for gen in range(1, GENERATIONS + 1):
        print(f"\n=== GEN {gen}/{GENERATIONS} | parent score={best_score:.3f} ===")
        candidates: list[dict[str, Any]] = []
        # candidat 0 = champion inchangé pour élitisme
        elite = json.loads(json.dumps(best_agent))
        elite["run_id"] = f"g{gen:04d}_c00_elite_{random.randint(0,999999):06d}"
        elite["generation"] = gen
        elite["candidate"] = 0
        candidates.append(elite)
        for ci in range(1, POPULATION):
            candidates.append(mutate_agent(best_agent["actions"], gen, ci))

        gen_best_agent = best_agent
        gen_best_result = best_result
        gen_best_score = best_score

        for cand in candidates:
            result = evaluate(cand, reload_lua=False)
            score = float(result.get("score", -999999.0))
            save_history(result)
            print(
                f"g{gen:03d} c{cand['candidate']:02d} "
                f"score={score:9.2f} dy={float(result.get('dy',0)):7.3f} "
                f"bestY={float(result.get('best_dy',0)):7.3f} above={float(result.get('above_knees',0)):6.2f} "
                f"ok={result.get('motor_ok')} timeout={result.get('timeout')}"
            )
            if score > gen_best_score:
                gen_best_score = score
                gen_best_agent = cand
                gen_best_result = result

        if gen_best_score > best_score:
            best_score = gen_best_score
            best_agent = gen_best_agent
            best_result = gen_best_result
            CHAMPION_AGENT.write_text(json.dumps(best_agent, indent=2), encoding="utf-8")
            write_rpl_from_actions(header, best_agent["actions"], CHAMPION_RPL, "xioi_master_final_v3_champion")
            shutil.copy2(CHAMPION_RPL, TORIBASH_REPLAY_DIR / CHAMPION_RPL.name)
            print("NEW CHAMPION:", best_score, CHAMPION_RPL)
        else:
            print("No improvement this generation.")

    print("\nDone.")
    print("Champion agent:", CHAMPION_AGENT)
    print("Champion RPL:", CHAMPION_RPL)


if __name__ == "__main__":
    main()
