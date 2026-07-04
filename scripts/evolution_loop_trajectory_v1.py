#!/usr/bin/env python3
import json
import random
import time
import subprocess
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
STEAM = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
SCRIPT_DIR = STEAM / "data" / "script"

CHAMPION = ROOT / "evolution" / "trajectory_champion_v1.json"
STATE = ROOT / "evolution" / "trajectory_v1_state.json"
POP_DIR = ROOT / "evolution" / "population_trajectory_v1"
BEST_DIR = ROOT / "evolution" / "best_trajectory_v1"

CANDIDATE_LUA = SCRIPT_DIR / "toribashai_candidate_actions_v1.lua"
RESULT_JSON = SCRIPT_DIR / "toribash_trajectory_result_v1.json"
RESET_COMMAND = "/lm ToribashAI/toribashai_xioi_city_v1.tbm"

POP_SIZE = 20
GENERATIONS = 999999
ELITES = 4
MAX_FRAME = 420
FRAME_STEP = 5
EVAL_WAIT = 90.0

JOINTS = list(range(20))
VALUES = [1, 2, 3, 4]

IMPORTANT_JOINTS = [4, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 19]

MUTATE_PAIR_RATE = 0.08
ADD_COMMAND_RATE = 0.25
DROP_COMMAND_RATE = 0.08
MOVE_COMMAND_RATE = 0.15
ADD_PAIR_RATE = 0.20
DROP_PAIR_RATE = 0.08

POP_DIR.mkdir(parents=True, exist_ok=True)
BEST_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def normalize(agent):
    out = {
        "name": agent.get("name", "trajectory_agent"),
        "segment": agent.get("segment", "launch_transition"),
        "max_frame": int(agent.get("max_frame", MAX_FRAME)),
        "commands": [],
    }

    for cmd in agent.get("commands", []):
        frame = int(cmd.get("frame", 0))
        frame = max(0, min(MAX_FRAME, frame))
        frame = (frame // FRAME_STEP) * FRAME_STEP

        pairs = []
        seen = set()

        for pair in cmd.get("pairs", []):
            if len(pair) != 2:
                continue
            j = int(pair[0])
            v = int(pair[1])
            if 0 <= j < 20 and 1 <= v <= 4 and j not in seen:
                pairs.append([j, v])
                seen.add(j)

        if pairs:
            out["commands"].append({"frame": frame, "pairs": pairs})

    out["commands"].sort(key=lambda c: c["frame"])
    return out


def commands_to_actions(agent):
    actions = []
    current = [3] * 20
    commands = normalize(agent)["commands"]

    by_frame = {}
    for cmd in commands:
        by_frame.setdefault(cmd["frame"], []).extend(cmd["pairs"])

    for frame in range(0, MAX_FRAME + 1, FRAME_STEP):
        for j, v in by_frame.get(frame, []):
            current[j] = v
        actions.append(current[:])

    return actions


def export_lua(agent):
    actions = commands_to_actions(agent)

    lines = ["candidate_actions = {"]
    for a in actions:
        lines.append("  {" + ", ".join(map(str, a)) + "},")
    lines.append("}")

    CANDIDATE_LUA.write_text("\n".join(lines), encoding="utf-8")


def reset_toribash():
    try:
        subprocess.run(["xdotool", "search", "--name", "Toribash", "windowactivate", "--sync"], check=False)
        time.sleep(0.10)

        try:
            p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
            p.communicate(RESET_COMMAND.encode("utf-8"))

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
            subprocess.run(["xdotool", "type", "--delay", "1", RESET_COMMAND], check=False)
            subprocess.run(["xdotool", "key", "Return"], check=False)

        print("Reset envoyé:", RESET_COMMAND)
    except Exception as e:
        print("Reset erreur:", e)

def read_result():
    if not RESULT_JSON.exists():
        return {"score": -999999.0, "reason": "no_result"}

    try:
        return json.loads(RESULT_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {"score": -999999.0, "reason": "bad_json"}


def wait_result(timeout=EVAL_WAIT):
    start = time.time()
    while time.time() - start < timeout:
        if RESULT_JSON.exists():
            try:
                result = read_result()
                RESULT_JSON.unlink(missing_ok=True)
                return result
            except Exception:
                pass
        time.sleep(0.25)

    return {"score": -999999.0, "reason": "timeout", "frames": 0}


def evaluate(agent):
    if RESULT_JSON.exists():
        RESULT_JSON.unlink()

    export_lua(agent)
    reset_toribash()

    result = wait_result()
    return float(result.get("score", -999999.0)), result


def random_pair():
    return [random.choice(IMPORTANT_JOINTS), random.choice(VALUES)]


def mutate(agent):
    child = normalize(agent)
    child["name"] = "trajectory_child_v1"

    commands = [json.loads(json.dumps(c)) for c in child["commands"]]

    # mutate existing commands
    new_commands = []
    for cmd in commands:
        if random.random() < DROP_COMMAND_RATE and len(commands) > 1:
            continue

        if random.random() < MOVE_COMMAND_RATE:
            delta = random.choice([-80, -40, -20, -10, 10, 20, 40, 80])
            cmd["frame"] = max(0, min(MAX_FRAME, cmd["frame"] + delta))
            cmd["frame"] = (cmd["frame"] // FRAME_STEP) * FRAME_STEP

        pairs = []
        for j, v in cmd["pairs"]:
            if random.random() < DROP_PAIR_RATE and len(cmd["pairs"]) > 1:
                continue

            if random.random() < MUTATE_PAIR_RATE:
                if random.random() < 0.65:
                    v = random.choice(VALUES)
                else:
                    j = random.choice(IMPORTANT_JOINTS)

            pairs.append([j, v])

        if random.random() < ADD_PAIR_RATE:
            pairs.append(random_pair())

        # dedupe pairs by joint
        dedup = {}
        for j, v in pairs:
            dedup[int(j)] = int(v)

        cmd["pairs"] = [[j, v] for j, v in dedup.items()]
        new_commands.append(cmd)

    commands = new_commands

    if random.random() < ADD_COMMAND_RATE:
        frame = random.randrange(0, MAX_FRAME + 1, FRAME_STEP)
        n_pairs = random.randint(2, 6)
        pairs = []
        used = set()
        for _ in range(n_pairs):
            j = random.choice(IMPORTANT_JOINTS)
            if j in used:
                continue
            used.add(j)
            pairs.append([j, random.choice(VALUES)])
        commands.append({"frame": frame, "pairs": pairs})

    child["commands"] = commands
    return normalize(child)


def make_population(champion):
    pop = [normalize(champion)]
    while len(pop) < POP_SIZE:
        pop.append(mutate(champion))
    return pop


def save_best(agent, score, result, gen, idx):
    payload = normalize(agent)
    payload["score"] = score
    payload["result"] = result
    payload["generation"] = gen
    payload["candidate"] = idx

    save_json(CHAMPION, payload)
    save_json(BEST_DIR / f"g{gen:04d}_c{idx:02d}_score_{score:.4f}.json", payload)


def load_state():
    if STATE.exists():
        try:
            return load_json(STATE)
        except Exception:
            pass
    return {"generation": 0, "best_score": -999999.0}


def main():
    print("=== ToribashAI Trajectory Evolution V1 ===")
    print(f"Champion: {CHAMPION}")
    print(f"Result:   {RESULT_JSON}")
    print("Make sure Toribash has:")
    print("  /ls toribash_trajectory_scorer_v1.lua")
    print("  /lm Urban_Structure/assassincreedhunter.tbm")

    champion = normalize(load_json(CHAMPION))
    state = load_state()
    best_score = float(state.get("best_score", -999999.0))
    start_gen = int(state.get("generation", 0)) + 1

    print(f"Starting generation: {start_gen}")
    print(f"Best score: {best_score:.6f}")

    print()
    print("=" * 49)
    print("INIT MANUEL TORIBASH")
    print("=" * 49)
    print("Dans Toribash lance :")
    print("/lm ToribashAI/toribashai_xioi_city_v1.tbm")
    print("/ls toribash_trajectory_scorer_v1.lua")
    input("\nQuand c'est prêt, appuie Entrée ici... ")

    for gen in range(start_gen, GENERATIONS + 1):
        print(f"\n--- Generation {gen} ---")

        population = make_population(champion)
        scored = []

        for idx, agent in enumerate(population):
            score, result = evaluate(agent)
            scored.append((score, agent, result))

            save_json(POP_DIR / f"g{gen:04d}_c{idx:02d}.json", {
                **normalize(agent),
                "score": score,
                "result": result,
            })

            print(
                f"g{gen:04d} c{idx:02d} | "
                f"score={score:.6f} | "
                f"avg_error={result.get('avg_error')} | "
                f"reason={result.get('reason')} | "
                f"frames={result.get('frames')}"
            )

            if score > best_score:
                best_score = score
                champion = normalize(agent)
                save_best(champion, best_score, result, gen, idx)
                print(f"💜 NEW BEST {best_score:.6f}")

        scored.sort(key=lambda x: x[0], reverse=True)

        # next champion from best candidate this generation if better than current champion candidate
        if scored[0][0] >= best_score:
            champion = normalize(scored[0][1])

        save_json(STATE, {
            "generation": gen,
            "best_score": best_score,
        })

        print(f"Best so far: {best_score:.6f}")
        print(f"Gen best:    {scored[0][0]:.6f}")


if __name__ == "__main__":
    main()
