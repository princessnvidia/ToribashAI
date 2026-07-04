#!/usr/bin/env python3
import copy
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"

SEED_AGENT = ROOT / "evolution/walk_xioi_imitation_seed_v1.json"
CHAMPION_AGENT = ROOT / "evolution/walk_xioi_imitation_champion_v1.json"

TORIBASH_SCRIPT_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"
)

PROJECT_LUA = ROOT / "scripts/toribash_walk_xioi_imitation_runner_v2.lua"
TORIBASH_LUA = TORIBASH_SCRIPT_DIR / "toribash_walk_xioi_imitation_runner_v2.lua"
TORIBASH_JSON = TORIBASH_SCRIPT_DIR / "walk_xioi_imitation_champion_v1.json"
RESULT_PATH = TORIBASH_SCRIPT_DIR / "toribashai_episode_result.json"

POP_DIR = ROOT / "evolution/population_walk_xioi_imitation_v2"
BEST_DIR = ROOT / "evolution/best_walk_xioi_imitation_v2"
STATE_PATH = ROOT / "evolution/walk_xioi_imitation_v2_state.json"

POP_SIZE = 12
GENERATIONS = 999999

JOINT_VALUES = [1, 2, 3, 4]

# Imitation = mutations plus douces au début
BASE_MUTATION_RATE = 0.018
PAIR_ADD_RATE = 0.006
PAIR_DROP_RATE = 0.004

EARLY_FRAME_PROTECTION = 80
EARLY_MUTATION_FACTOR = 0.20

RESET_COMMAND = "/lm ToribashAI/toribashai_goal_flat_v1.tbm"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2), encoding="utf-8")


def ensure_seed():
    if CHAMPION_AGENT.exists():
        return

    if not SEED_AGENT.exists():
        raise FileNotFoundError(SEED_AGENT)

    shutil.copy2(SEED_AGENT, CHAMPION_AGENT)
    print("Champion imitation initial créé depuis seed:", CHAMPION_AGENT)


def normalize_agent(agent):
    agent = copy.deepcopy(agent)
    agent["branch"] = "walk_xioi_imitation"
    agent["loop_length"] = int(agent.get("loop_length", 428))

    commands = agent.get("commands", [])
    clean = []

    for cmd in commands:
        if not isinstance(cmd, dict):
            continue

        frame = int(cmd.get("frame", 0))
        pairs = cmd.get("pairs", [])

        new_pairs = []
        seen = set()

        for pair in pairs:
            if not isinstance(pair, list) or len(pair) < 2:
                continue

            joint = int(pair[0])
            state = int(pair[1])

            if 0 <= joint <= 19 and 1 <= state <= 4 and joint not in seen:
                new_pairs.append([joint, state])
                seen.add(joint)

        clean.append({"frame": frame, "pairs": new_pairs})

    clean.sort(key=lambda c: c["frame"])
    agent["commands"] = clean

    return agent


def mutation_rate_for_frame(frame):
    if frame < EARLY_FRAME_PROTECTION:
        return BASE_MUTATION_RATE * EARLY_MUTATION_FACTOR
    return BASE_MUTATION_RATE


def mutate(agent):
    child = normalize_agent(agent)

    mutations = 0
    added = 0
    dropped = 0

    loop_length = int(child.get("loop_length", 428))

    for cmd in child["commands"]:
        frame = int(cmd["frame"])
        rate = mutation_rate_for_frame(frame)

        new_pairs = []

        for joint, state in cmd["pairs"]:
            if random.random() < PAIR_DROP_RATE and frame >= EARLY_FRAME_PROTECTION:
                dropped += 1
                mutations += 1
                continue

            if random.random() < rate:
                choices = [v for v in JOINT_VALUES if v != int(state)]
                state = random.choice(choices)
                mutations += 1

            new_pairs.append([int(joint), int(state)])

        if random.random() < PAIR_ADD_RATE and frame >= EARLY_FRAME_PROTECTION:
            existing = {p[0] for p in new_pairs}
            possible = [j for j in range(20) if j not in existing]
            if possible:
                new_pairs.append([random.choice(possible), random.choice(JOINT_VALUES)])
                added += 1
                mutations += 1

        cmd["pairs"] = new_pairs

    if random.random() < 0.04:
        frame = random.randint(EARLY_FRAME_PROTECTION, loop_length - 1)
        joint = random.randint(0, 19)
        state = random.choice(JOINT_VALUES)
        child["commands"].append({"frame": frame, "pairs": [[joint, state]]})
        added += 1
        mutations += 1

    child["commands"].sort(key=lambda c: c["frame"])

    child["mutations"] = mutations
    child["pairs_added"] = added
    child["pairs_dropped"] = dropped
    child["mutation_rate"] = BASE_MUTATION_RATE
    child["mutation_strategy"] = "walk_xioi_imitation_v2_soft_mutation"

    return child


def copy_lua_to_steam():
    shutil.copy2(PROJECT_LUA, TORIBASH_LUA)
    print("Lua copié:", TORIBASH_LUA)


def export_candidate(agent):
    agent = normalize_agent(agent)
    save_json(CHAMPION_AGENT, agent)
    shutil.copy2(CHAMPION_AGENT, TORIBASH_JSON)
    print("JSON copié:", TORIBASH_JSON)


def focus_toribash():
    subprocess.run(
        ["xdotool", "search", "--name", "Toribash", "windowactivate", "--sync"],
        check=False,
    )
    time.sleep(0.10)


def send_chat_command(command):
    focus_toribash()

    try:
        p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
        p.communicate(command.encode("utf-8"))

        subprocess.run(["xdotool", "key", "t"], check=False)
        time.sleep(0.08)

        subprocess.run(["xdotool", "key", "ctrl+a"], check=False)
        time.sleep(0.03)

        subprocess.run(["xdotool", "key", "BackSpace"], check=False)
        time.sleep(0.03)

        subprocess.run(["xdotool", "key", "ctrl+v"], check=False)
        time.sleep(0.03)

        subprocess.run(["xdotool", "key", "Return"], check=False)

    except Exception:
        subprocess.run(["xdotool", "key", "t"], check=False)
        time.sleep(0.08)
        subprocess.run(["xdotool", "type", "--delay", "1", command], check=False)
        subprocess.run(["xdotool", "key", "Return"], check=False)


def reset_toribash():
    RESULT_PATH.unlink(missing_ok=True)
    send_chat_command(RESET_COMMAND)
    time.sleep(0.8)


def reload_lua():
    send_chat_command("/ls toribash_walk_xioi_imitation_runner_v2.lua")
    time.sleep(0.5)


def wait_result(timeout=45):
    start = time.time()

    while time.time() - start < timeout:
        if RESULT_PATH.exists():
            try:
                result = load_json(RESULT_PATH)
                RESULT_PATH.unlink(missing_ok=True)
                return result
            except Exception:
                pass

        time.sleep(0.20)

    return {
        "score": -999999,
        "reason": "timeout",
        "frames": 0,
    }


def evaluate(agent, gen, idx):
    agent = normalize_agent(agent)
    agent["name"] = f"walk_xioi_imitation_v2_gen_{gen:05d}_agent_{idx:03d}"
    agent["current_generation"] = gen
    agent["current_candidate"] = idx
    agent["population_size"] = POP_SIZE

    candidate_path = POP_DIR / f"gen_{gen:05d}_agent_{idx:03d}.json"
    save_json(candidate_path, agent)

    export_candidate(agent)

    print()
    print("=" * 50)
    print(f"GEN {gen} | CANDIDAT {idx:03d}")
    print("=" * 50)
    print("Agent:", agent["name"])
    print("Mutations:", agent.get("mutations", 0))
    print("Added:", agent.get("pairs_added", 0))
    print("Dropped:", agent.get("pairs_dropped", 0))

    reset_toribash()
    reload_lua()

    result = wait_result()
    score = float(result.get("score", -999999))

    result_path = POP_DIR / f"gen_{gen:05d}_agent_{idx:03d}_result.json"
    save_json(result_path, result)

    print("SCORE =", score)
    print("RESULT =", result)

    return score, result, candidate_path


def main():
    POP_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)

    ensure_seed()
    copy_lua_to_steam()

    champion = normalize_agent(load_json(CHAMPION_AGENT))
    champion_score = -999999

    print()
    print("=" * 50)
    print("WALK XIOI IMITATION V2")
    print("=" * 50)
    print("Dans Toribash, laisse la fenêtre ouverte.")
    input("Appuie Entrée quand Toribash est prêt... ")

    for gen in range(GENERATIONS):
        print()
        print("=" * 50)
        print(f"GENERATION {gen}")
        print("=" * 50)

        candidates = []

        for i in range(POP_SIZE):
            if i == 0:
                elite = normalize_agent(champion)
                elite["elite_copy"] = True
                elite["mutations"] = 0
                elite["pairs_added"] = 0
                elite["pairs_dropped"] = 0
                candidates.append(elite)
            else:
                child = mutate(champion)
                child["elite_copy"] = False
                candidates.append(child)

        best_score = -999999
        best_result = None
        best_path = None

        for i, candidate in enumerate(candidates):
            score, result, path = evaluate(candidate, gen, i)

            if score > best_score:
                best_score = score
                best_result = result
                best_path = path

        print()
        print("=" * 50)
        print("MEILLEUR CANDIDAT")
        print("=" * 50)
        print("Score:", best_score)
        print("Path:", best_path)
        print("Result:", best_result)

        if best_score > champion_score:
            champion_score = best_score
            champion = normalize_agent(load_json(best_path))

            champion_save = BEST_DIR / f"champion_gen_{gen:05d}_score_{champion_score:.2f}.json"
            save_json(champion_save, champion)
            shutil.copy2(champion_save, CHAMPION_AGENT)

            print("NOUVEAU CHAMPION:", champion_save)
        else:
            print("Pas mieux que le champion actuel")

        save_json(
            STATE_PATH,
            {
                "generation": gen,
                "champion_score": champion_score,
                "champion_path": str(CHAMPION_AGENT),
                "best_score_this_gen": best_score,
                "best_result_this_gen": best_result,
                "mutation_rate": BASE_MUTATION_RATE,
                "mutation_strategy": "walk_xioi_imitation_v2_soft_mutation",
                "mod": "ToribashAI/toribashai_goal_flat_v1.tbm",
                "population": POP_SIZE,
            },
        )


if __name__ == "__main__":
    main()
