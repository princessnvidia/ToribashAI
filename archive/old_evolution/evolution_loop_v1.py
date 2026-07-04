#!/usr/bin/env python3
from pathlib import Path
import json
import random
import time
import shutil

ROOT = Path.home() / "Documents" / "ToribashAI"

AGENT_SRC = ROOT / "toribashai_agent_current.lua"
AGENT_ACTIVE = ROOT / "lua_runtime" / "toribashai_agent_candidate.lua"

RESULT_FILE = ROOT / "lua_runtime" / "episode_result.json"
CANDIDATES_DIR = ROOT / "evolution" / "candidates"
BEST_DIR = ROOT / "evolution" / "best"

POPULATION = 12
GENERATIONS = 999999
MUTATION_RATE = 0.12
MUTATION_STRENGTH = 1

JOINT_VALUES = [1, 2, 3, 4]

CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
BEST_DIR.mkdir(parents=True, exist_ok=True)
AGENT_ACTIVE.parent.mkdir(parents=True, exist_ok=True)


def load_agent_text():
    return AGENT_SRC.read_text(encoding="utf-8")


def mutate_lua_numbers(text):
    out = []
    i = 0

    while i < len(text):
        c = text[i]

        if c.isdigit() and random.random() < MUTATION_RATE:
            old = int(c)
            if old in JOINT_VALUES:
                new = random.choice(JOINT_VALUES)
                out.append(str(new))
            else:
                out.append(c)
            i += 1
        else:
            out.append(c)
            i += 1

    return "".join(out)


def save_candidate(text, gen, idx):
    path = CANDIDATES_DIR / f"gen_{gen:05d}_candidate_{idx:03d}.lua"
    path.write_text(text, encoding="utf-8")
    return path


def wait_for_result(timeout=60):
    start = time.time()

    while time.time() - start < timeout:
        if RESULT_FILE.exists():
            try:
                data = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
                RESULT_FILE.unlink()
                return data
            except Exception:
                pass

        time.sleep(0.25)

    return {
        "score": -999999,
        "error": "timeout"
    }


def evaluate_candidate(candidate_path):
    shutil.copy(candidate_path, AGENT_ACTIVE)

    print()
    print("================================================")
    print("CANDIDAT CHARGÉ DANS TORIBASH")
    print(f"Fichier : {candidate_path.name}")
    print()
    print("Dans Toribash, lance/reload le runner Lua.")
    print("Quand l'épisode finit, le Lua doit écrire :")
    print(RESULT_FILE)
    print("================================================")

    result = wait_for_result(timeout=120)

    score = float(result.get("score", -999999))
    return score, result


def main():
    base = load_agent_text()
    champion = base
    champion_score = -999999

    for gen in range(GENERATIONS):
        print()
        print(f"========== GENERATION {gen} ==========")

        scored = []

        for i in range(POPULATION):
            if gen == 0 and i == 0:
                candidate = champion
            else:
                candidate = mutate_lua_numbers(champion)

            candidate_path = save_candidate(candidate, gen, i)

            score, result = evaluate_candidate(candidate_path)

            print(f"Score candidat {i}: {score}")
            print(result)

            scored.append((score, candidate_path, result))

        scored.sort(key=lambda x: x[0], reverse=True)

        best_score, best_path, best_result = scored[0]

        print()
        print("MEILLEUR DE LA GENERATION")
        print("Score:", best_score)
        print("Fichier:", best_path)

        if best_score > champion_score:
            champion_score = best_score
            champion = best_path.read_text(encoding="utf-8")

            champion_path = BEST_DIR / f"champion_gen_{gen:05d}_score_{champion_score:.2f}.lua"
            champion_path.write_text(champion, encoding="utf-8")

            shutil.copy(champion_path, AGENT_SRC)

            print()
            print("NOUVEAU CHAMPION SAUVÉ")
            print(champion_path)

        else:
            print("Pas mieux que le champion actuel.")

        summary_path = ROOT / "evolution" / "last_generation_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "generation": gen,
                    "champion_score": champion_score,
                    "best_score_this_gen": best_score,
                    "best_result": best_result,
                    "scores": [s for s, _, _ in scored],
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
