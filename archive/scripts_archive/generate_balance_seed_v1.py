#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"

TORIBASH_SCRIPT_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"
)

OUT_JSON = ROOT / "evolution/balance_seed_v1.json"
OUT_LUA = TORIBASH_SCRIPT_DIR / "toribashai_agent_current.lua"

CONTROL_JOINTS = list(range(20))

# Pose debout prudente.
BASE_ACTION = [
    3, 3, 3, 1,
    3, 3, 3, 3,
    3, 3,
    3, 3, 3, 3,
    3, 3, 3, 3, 3, 3,
]

NUM_ACTIONS = 20


def export_lua(agent):
    lines = []
    lines.append("-- Auto-generated balance seed")
    lines.append("TORIBASHAI_AGENT = {}")
    lines.append(f'TORIBASHAI_AGENT.name = "{agent["name"]}"')
    lines.append("TORIBASHAI_AGENT.control_joints = { " + ", ".join(str(j) for j in CONTROL_JOINTS) + " }")
    lines.append("TORIBASHAI_AGENT.actions = {")

    for action in agent["actions"]:
        lines.append("    { " + ", ".join(str(int(v)) for v in action) + " },")

    lines.append("}")
    lines.append("return TORIBASHAI_AGENT")

    OUT_LUA.write_text("\n".join(lines), encoding="utf-8")


def main():
    agent = {
        "name": "balance_seed_v1",
        "skill": "balance",
        "control_joints": CONTROL_JOINTS,
        "actions": [BASE_ACTION[:] for _ in range(NUM_ACTIONS)],
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(agent, indent=2), encoding="utf-8")
    export_lua(agent)

    print("Balance seed exported")
    print("JSON:", OUT_JSON)
    print("Lua :", OUT_LUA)


if __name__ == "__main__":
    main()
