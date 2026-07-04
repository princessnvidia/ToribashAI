#!/usr/bin/env python3

from pathlib import Path
import copy
import json
import random
import shutil
import subprocess
import time

ROOT = Path.home() / "Documents" / "ToribashAI"

SEED_AGENT = ROOT / "evolution" / "walk_fullbody_agent_current.json"

POP_DIR = ROOT / "evolution" / "population_fullbody_v7_lua_reset"
BEST_DIR = ROOT / "evolution" / "best_fullbody_v7_lua_reset"

TORIBASH_SCRIPT_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"
)

RESULT_FILE = TORIBASH_SCRIPT_DIR / "toribashai_episode_result.json"
CONTROL_FILE = TORIBASH_SCRIPT_DIR / "toribashai_control_v23.txt"
EXPORT_SCRIPT = ROOT / "scripts" / "export_agent_lua_v1.py"

POP_SIZE = 20
GENERATIONS = 999999
MUTATION_RATE = 0.02
KEEP_ONLY_BEST = True

JOINT_VALUES = [1, 2, 3, 4]

POP_DIR.mkdir(parents=True, exist_ok=True)
BEST_DIR.mkdir(parents=True, exist_ok=True)
RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def mutate_value(v):
    if isinstance(v, int) and v in JOINT_VALUES:
        if random.random() < MUTATION_RATE:
            return random.choice(JOINT_VALUES)

    if isinstance(v, list):
        return [mutate_value(x) for x in v]

    if isinstance(v, dict):
        return {k: mutate_value(x) for k, x in v.items()}

    return v


def export_to_lua(agent_path):
    subprocess.run(
        ["python3", str(EXPORT_SCRIPT), str(agent_path)],
        cwd=str(ROOT),
        check=True,
    )


def request_reset():
    CONTROL_FILE.write_text("reset\n", encoding="utf-8")
    time.sleep(0.5)


def wait_result(timeout=180):
    start = time.time()

    while time.time() - start < timeout:
        if RESULT_FILE.exists():
            try:
                data = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
                RESULT_FILE.unlink()
                return data
            except Exception:
                time.sleep(0.25)

        time.sleep(0.25)

    return {
        "score": -999999,
        "error": "timeout",
    }


def evaluate(agent_path, gen, i):
    if RESULT_FILE.exists():
        RESULT_FILE.unlink()

    export_to_lua(agent_path)

    print()
    print("=================================================")
    print(f"GEN {gen} | CANDIDAT {i:03d}")
    print("Agent exporté :", agent_path.name)
    print("Reset demandé via fichier Lua :", CONTROL_FILE)
    print("=================================================")

    request_reset()

    result = wait_result()
    score = float(result.get("score", -999999))

    return score, result


def cleanup_generation(scored):
    if not KEEP_ONLY_BEST:
        return

    for _, path, _ in scored[1:]:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def main():
    if not SEED_AGENT.exists():
        raise FileNotFoundError(f"Seed introuvable : {SEED_AGENT}")

    print()
    print("=================================================")
    print("INIT TORIBASH")
    print("=================================================")
    print("Dans Toribash, lance une seule fois :")
    print("/lm toribashai_goal_flat_v1.tbm")
    print("/ls toribash_reward_runner_v23.lua")
    print()
    input("Quand V23 est chargé, appuie Entrée ici... ")

    champion = load_json(SEED_AGENT)
    champion_score = -999999

    for gen in range(GENERATIONS):
        print()
        print("=================================================")
        print(f"GENERATION {gen}")
        print("=================================================")

        scored = []

        for i in range(POP_SIZE):
            if i == 0:
                candidate = copy.deepcopy(champion)
            else:
                candidate = mutate_value(copy.deepcopy(champion))

            candidate["name"] = f"agent_{i:03d}"
            candidate["generation"] = gen
            candidate["candidate_id"] = i
            candidate["mutation_rate"] = MUTATION_RATE
            candidate["runner"] = "toribash_reward_runner_v23.lua"

            path = POP_DIR / f"gen_{gen:05d}_agent_{i:03d}.json"
            save_json(path, candidate)

            score, result = evaluate(path, gen, i)

            print()
            print(f"CANDIDAT {i:03d}")
            print(f"SCORE = {score:.4f}")
            print(result)

            scored.append((score, path, result))

        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_path, best_result = scored[0]

        print()
        print("=================================================")
        print("MEILLEUR CANDIDAT")
        print("=================================================")
        print("Score :", best_score)
        print("Fichier :", best_path)

        if best_score > champion_score:
            champion_score = best_score
            champion = load_json(best_path)

            champion_path = (
                BEST_DIR
                / f"champion_gen_{gen:05d}_score_{champion_score:.2f}.json"
            )

            save_json(champion_path, champion)
            shutil.copy(champion_path, SEED_AGENT)

            print()
            print("NOUVEAU CHAMPION")
            print(champion_path)
        else:
            print()
            print("Pas mieux que le champion actuel")

        cleanup_generation(scored)

        summary = {
            "generation": gen,
            "champion_score": champion_score,
            "best_score_this_generation": best_score,
            "best_path": str(best_path),
            "best_result": best_result,
            "scores": [x[0] for x in scored],
            "mutation_rate": MUTATION_RATE,
            "population": POP_SIZE,
            "runner": "toribash_reward_runner_v23.lua",
        }

        save_json(
            ROOT / "evolution" / "fullbody_v7_lua_reset_last_summary.json",
            summary,
        )


if __name__ == "__main__":
    main()
