#!/usr/bin/env python3
import json
import random
import time
import subprocess
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
STEAM_TB = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
SCRIPT_DIR = STEAM_TB / "data" / "script"

ACTIONS_LUA = SCRIPT_DIR / "toribashai_launch_actions_v8.lua"
RESULT_JSON = SCRIPT_DIR / "toribashai_launch_result_v8.json"
EPOCH_TXT = SCRIPT_DIR / "toribashai_launch_epoch_v8.txt"

EV_DIR = ROOT / "evolution" / "launch_v8"
EV_DIR.mkdir(parents=True, exist_ok=True)

BEST_JSON = EV_DIR / "best_launch_v8.json"
POP_SIZE = 20
GENERATIONS = 50
ACTION_LEN = 48
MUTATION_RATE = 0.04
EVAL_WAIT = 4.0

JOINTS = list(range(20))
VALUES = [1, 2, 3, 4]

# Launch structuré :
# phase 1 = compression
# phase 2 = poussée jambe arrière
# phase 3 = contrepoids bras/torse
# phase 4 = stabilisation avant walk
BASE_ACTIONS = [
    # compression / crouch stable
    [3,3,3,3, 2,2,3,3, 3,3,3,3, 3,3, 2,2,4,4, 3,3],
    [3,3,3,3, 2,2,3,3, 3,3,3,3, 3,3, 2,2,4,4, 3,3],
    [3,3,3,3, 2,2,3,3, 3,3,3,3, 3,3, 2,2,4,4, 3,3],

    # push asymétrique : une jambe pousse, l'autre reçoit
    [3,3,3,3, 1,2,3,2, 2,1,3,3, 3,3, 4,1,1,4, 2,3],
    [3,3,3,3, 1,2,3,2, 2,1,3,3, 3,3, 4,1,1,4, 2,3],
    [3,3,3,3, 1,2,3,2, 2,1,3,3, 3,3, 4,1,1,4, 2,3],

    # transfert poids / bras contrebalancent
    [3,3,3,3, 2,1,3,1, 1,2,3,3, 3,3, 1,4,4,1, 3,2],
    [3,3,3,3, 2,1,3,1, 1,2,3,3, 3,3, 1,4,4,1, 3,2],
    [3,3,3,3, 2,1,3,1, 1,2,3,3, 3,3, 1,4,4,1, 3,2],

    # stabilisation dynamique
    [3,3,3,3, 3,3,3,3, 2,2,3,3, 3,3, 3,3,3,3, 3,3],
    [3,3,3,3, 3,3,3,3, 2,2,3,3, 3,3, 3,3,3,3, 3,3],
]

def expand_seed():
    actions = []
    while len(actions) < ACTION_LEN:
        for a in BASE_ACTIONS:
            actions.append(a[:])
            if len(actions) >= ACTION_LEN:
                break
    return actions


def random_agent():
    actions = expand_seed()
    return mutate(actions, rate=0.35)


def mutate(actions, rate=MUTATION_RATE):
    child = [a[:] for a in actions]

    # Joints importants pour launch :
    # 4-7 pecs/chest/lumbar/abs, 14-19 hanches/genoux/chevilles
    core = [4, 5, 6, 7]
    legs = [14, 15, 16, 17, 18, 19]
    arms = [8, 9, 10, 11]
    important = core + legs

    for i in range(len(child)):
        # début plus mutable que fin
        if i < 18:
            phase_mult = 2.0
        elif i < 34:
            phase_mult = 1.4
        else:
            phase_mult = 0.7

        # jambes/core mutent plus souvent, bras moins
        for j in important:
            if random.random() < rate * phase_mult * 2.2:
                child[i][j] = random.choice(VALUES)

        for j in arms:
            if random.random() < rate * phase_mult * 0.8:
                child[i][j] = random.choice(VALUES)

        # garde le reste relativement stable
        for j in JOINTS:
            if j not in important and j not in arms:
                if random.random() < rate * phase_mult * 0.25:
                    child[i][j] = random.choice(VALUES)

        # mutation de segment : répéter une bonne phase
        if random.random() < rate * 0.15:
            src = random.randrange(len(child))
            child[i] = child[src][:]

    return child

def write_lua(actions):
    lines = ["launch_actions = {"]
    for a in actions:
        lines.append("  {" + ", ".join(map(str, a)) + "},")
    lines.append("}")
    ACTIONS_LUA.write_text("\n".join(lines), encoding="utf-8")


def reset_toribash():
    cmd = "/reset"
    subprocess.run(["xclip", "-selection", "clipboard"], input=cmd.encode(), check=False)

    subprocess.run(["xdotool", "key", "t"], check=False)
    time.sleep(0.08)
    subprocess.run(["xdotool", "key", "ctrl+a"], check=False)
    time.sleep(0.03)
    subprocess.run(["xdotool", "key", "BackSpace"], check=False)
    time.sleep(0.03)
    subprocess.run(["xdotool", "key", "ctrl+v"], check=False)
    time.sleep(0.03)
    subprocess.run(["xdotool", "key", "Return"], check=False)
    time.sleep(0.20)


def read_result():
    if not RESULT_JSON.exists():
        return {"score": -999999, "reason": "no_result"}

    try:
        return json.loads(RESULT_JSON.read_text())
    except Exception:
        return {"score": -999999, "reason": "bad_json"}


def evaluate(actions):
    if RESULT_JSON.exists():
        RESULT_JSON.unlink()

    write_lua(actions)
    EPOCH_TXT.write_text(str(time.time()), encoding="utf-8")
    reset_toribash()
    time.sleep(EVAL_WAIT)

    result = read_result()
    return float(result.get("score", -999999)), result


def save_best(actions, score, result, gen, idx):
    payload = {
        "name": "launch_v8_best",
        "generation": gen,
        "candidate": idx,
        "score": score,
        "result": result,
        "action_len": len(actions),
        "actions": actions,
    }
    BEST_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_best():
    if not BEST_JSON.exists():
        return None

    try:
        data = json.loads(BEST_JSON.read_text())
        return data["actions"], float(data["score"])
    except Exception:
        return None


def main():
    print("=== ToribashAI Launch Evolution V8 ===")
    print(f"ACTIONS: {ACTIONS_LUA}")
    print(f"RESULT:  {RESULT_JSON}")
    print(f"BEST:    {BEST_JSON}")

    loaded = load_best()
    if loaded:
        best_actions, best_score = loaded
        print(f"Loaded best score: {best_score:.3f}")
    else:
        best_actions = expand_seed()
        best_score = -999999.0

    population = [best_actions]
    while len(population) < POP_SIZE:
        population.append(mutate(best_actions, rate=0.25))

    for gen in range(1, GENERATIONS + 1):
        print(f"\n--- Generation {gen}/{GENERATIONS} ---")

        scored = []

        for idx, agent in enumerate(population):
            score, result = evaluate(agent)
            scored.append((score, agent, result))

            print(
                f"g{gen:03d} c{idx:02d} | "
                f"score={score:.3f} | "
                f"progress={result.get('progress_y')} | "
                f"head_z={result.get('head_z')} | "
                f"hip_z={result.get('hip_z')} | "
                f"reason={result.get('reason')}"
            )

            if score > best_score:
                best_score = score
                best_actions = agent
                save_best(best_actions, best_score, result, gen, idx)
                print(f"💜 NEW BEST: {best_score:.3f}")

        scored.sort(key=lambda x: x[0], reverse=True)
        elites = [scored[0][1], scored[1][1], best_actions]

        population = elites[:]
        while len(population) < POP_SIZE:
            parent = random.choice(elites)
            population.append(mutate(parent))

    print("\nTerminé 💜")
    print(f"Best score: {best_score:.3f}")
    print(f"Saved: {BEST_JSON}")


if __name__ == "__main__":
    main()
