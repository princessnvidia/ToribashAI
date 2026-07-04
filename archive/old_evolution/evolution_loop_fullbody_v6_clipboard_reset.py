#!/usr/bin/env python3

from pathlib import Path
import json
import random
import shutil
import subprocess
import time
import copy

ROOT = Path.home() / "Documents" / "ToribashAI"

SEED_AGENT = ROOT / "evolution" / "walk_fullbody_agent_current.json"

POP_DIR = ROOT / "evolution" / "population_fullbody_v6_clipboard"
BEST_DIR = ROOT / "evolution" / "best_fullbody_v6_clipboard"

RESULT_FILE = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script/toribashai_episode_result.json"
)

EXPORT_SCRIPT = ROOT / "scripts" / "export_agent_lua_v1.py"

POP_SIZE = 10
GENERATIONS = 999999

MUTATION_RATE = 0.05

JOINT_VALUES = [1, 2, 3, 4]

POP_DIR.mkdir(parents=True, exist_ok=True)
BEST_DIR.mkdir(parents=True, exist_ok=True)
RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def mutate_value(v):
    if isinstance(v, int) and v in JOINT_VALUES:
        if random.random() < MUTATION_RATE:
            return random.choice(JOINT_VALUES)

    if isinstance(v, list):
        return [mutate_value(x) for x in v]

    if isinstance(v, dict):
        return {k: mutate_value(x) for k, x in v.items()}

    return v

def eliminate_bad_behavior(candidate, result):
    # Élimine les plongeons / levier vers l’avant
    z = float(result.get("z", 999))
    progress_y = float(result.get("progress_y", 0))
    frames = int(result.get("frames", 0))

    # Si elle avance beaucoup mais finit trop basse = probablement jetée / levier
    if progress_y > 3.0 and z < 5.8:
        return True, "dive_forward"

    # Si elle avance très vite en peu de frames = exploit probable
    if frames < 120 and progress_y > 2.5:
        return True, "fast_throw"

    # Si Toribash détecte une chute
    if result.get("fell", False):
        return True, "fell"

    return False, ""


def export_to_lua(agent_path):
    subprocess.run(
        ["python3", str(EXPORT_SCRIPT), str(agent_path)],
        cwd=str(ROOT),
        check=True,
    )


def run_shell(cmd):
    subprocess.run(cmd, shell=True, check=True)


def send_reset_clipboard():
    run_shell('printf "/reset" | xclip -selection clipboard')
    run_shell('xdotool search --name "Toribash" windowactivate --sync')

    run_shell("xdotool key t")
    time.sleep(3)

    run_shell("xdotool key ctrl+a")
    time.sleep(0.2)

    run_shell("xdotool key BackSpace")
    time.sleep(0.2)

    run_shell("xdotool key ctrl+v")
    time.sleep(0.5)

    run_shell("xdotool key Return")


def wait_result(timeout=90):
    start = time.time()

    while time.time() - start < timeout:
        if RESULT_FILE.exists():
            try:
                data = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
                RESULT_FILE.unlink()
                return data
            except Exception:
                time.sleep(0.2)

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
    print("Agent exporté vers Toribash")
    print("Reset clipboard automatique")
    print("Attente :", RESULT_FILE)
    print("=================================================")

    send_reset_clipboard()

    result = wait_result()

    score = float(result.get("score", -999999))

    return score, result


def main():
    if not SEED_AGENT.exists():
        raise FileNotFoundError(
            f"Seed introuvable : {SEED_AGENT}"
        )

    print()
    print("=================================================")
    print("INIT MANUEL TORIBASH")
    print("=================================================")
    print("Dans Toribash, lance une seule fois :")
    print("/lm toribashai_goal_flat_v1.tbm")
    print("/ls toribash_reward_runner_v13_2.lua")
    print()
    input("Quand c'est prêt, appuie Entrée ici... ")

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
            candidate["runner"] = "toribash_reward_runner_v13.lua"

            path = POP_DIR / f"gen_{gen:05d}_agent_{i:03d}.json"
            save_json(path, candidate)

            score, result = evaluate(path, gen, i)

            bad, reason = eliminate_bad_behavior(candidate, result)
            if bad:
                print("CANDIDAT ÉLIMINÉ:", reason)
                score = -999999
                result["eliminated"] = True
                result["elimination_reason"] = reason

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

        summary = {
            "generation": gen,
            "champion_score": champion_score,
            "best_score_this_generation": best_score,
            "best_result": best_result,
            "best_path": str(best_path),
            "scores": [x[0] for x in scored],
            "mutation_rate": MUTATION_RATE,
            "population": POP_SIZE,
            "runner": "toribash_reward_runner_v13.lua",
        }

        save_json(
            ROOT / "evolution" / "fullbody_v6_clipboard_last_summary.json",
            summary,
        )


if __name__ == "__main__":
    main()
