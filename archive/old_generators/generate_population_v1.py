#!/usr/bin/env python3
import json
import random
from pathlib import Path

PROJECT = Path.home() / "Documents" / "ToribashAI"
EVOLUTION_DIR = PROJECT / "evolution"
POPULATION_DIR = EVOLUTION_DIR / "population"

NUM_AGENTS = 20
NUM_TURNS = 45
LEG_JOINTS = [14, 15, 16, 17, 18, 19]
JOINT_VALUES = [1, 2, 3, 4]

SEED = 42000


def make_agent(agent_index: int) -> dict:
    rng = random.Random(SEED + agent_index)

    actions = []

    for _ in range(NUM_TURNS):
        turn = [rng.choice(JOINT_VALUES) for _ in LEG_JOINTS]
        actions.append(turn)

    return {
        "version": 1,
        "name": f"agent_{agent_index:03d}",
        "agent_index": agent_index,
        "num_turns": NUM_TURNS,
        "leg_joints": LEG_JOINTS,
        "joint_values": JOINT_VALUES,
        "actions": actions,
        "score": None,
    }


def main():
    POPULATION_DIR.mkdir(parents=True, exist_ok=True)

    for old_file in POPULATION_DIR.glob("agent_*.json"):
        old_file.unlink()

    for i in range(NUM_AGENTS):
        agent = make_agent(i)
        path = POPULATION_DIR / f"agent_{i:03d}.json"

        path.write_text(
            json.dumps(agent, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print("Generated:", path)

    current = POPULATION_DIR / "agent_000.json"
    target = EVOLUTION_DIR / "agent_current.json"
    target.write_text(current.read_text(encoding="utf-8"), encoding="utf-8")

    print()
    print("Population ready.")
    print("Current agent:", target)


if __name__ == "__main__":
    main()
