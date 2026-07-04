#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime

ROOT = Path.home() / "Documents" / "ToribashAI"

OUT = ROOT / "evolution" / "multiskill_agent_v7.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

agent = {
    "name": "toribashai_multiskill_v7",
    "version": 7,
    "created_at": datetime.now().isoformat(timespec="seconds"),

    "principle": {
        "rpl": "full physics reference",
        "gru": "movement skill",
        "lua": "executes and scores",
        "controller": "selects skills",
    },

    "mod": "ToribashAI/toribashai_xioi_city_v1.tbm",

    "joints": list(range(20)),
    "seq_len": 8,

    "skills": {
        "launch": {
            "type": "gru",
            "model": str(ROOT / "models" / "skills" / "launch_gru_v1.pt"),
            "dataset": str(ROOT / "datasets" / "skills" / "launch_skill_v1.jsonl"),
            "description": "Creates initial walking inertia from Xioi frames 0→315.",
            "max_steps": 57
        },
        "walk": {
            "type": "gru",
            "model": str(ROOT / "models" / "skills" / "walk_gru_v1.pt"),
            "dataset": str(ROOT / "datasets" / "skills" / "walk_skill_v1.jsonl"),
            "description": "Maintains walking loop from Xioi len265 champion.",
            "max_steps": 46
        }
    },

    "schedule": [
        {
            "skill": "launch",
            "repeat": 1
        },
        {
            "skill": "walk",
            "repeat": "loop"
        }
    ],

    "recovery": {
        "enabled": False,
        "planned": True,
        "description": "Recovery skill will be connected later."
    }
}

OUT.write_text(json.dumps(agent, indent=2), encoding="utf-8")

print(f"[OK] Agent écrit: {OUT}")
