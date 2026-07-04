#!/usr/bin/env python3
"""
run_curated_walking_sampled_live_v25_1.py

V25.1:
- génère un flux d'actions depuis les vraies séquences curated walking V23.1
- génère aussi une table Lua native, pour éviter le parser JSON côté Lua
- copie le runner Lua côté Steam
- charge le runner dans Toribash
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
SCRIPTS = ROOT / "scripts"

TORIBASH_SCRIPT_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"
)

GENERATOR = SCRIPTS / "generate_sampled_walking_live_actions_v25_1.py"
PROJECT_LUA = SCRIPTS / "toribash_curated_walking_sampled_runner_v25_1.lua"
TORIBASH_LUA = TORIBASH_SCRIPT_DIR / "toribash_curated_walking_sampled_runner_v25_1.lua"

LUA_COMMAND = "/ls toribash_curated_walking_sampled_runner_v25_1.lua"
RESET_COMMAND = "/reset"


def focus_toribash() -> None:
    subprocess.run(["xdotool", "search", "--name", "Toribash", "windowactivate", "--sync"], check=False)
    time.sleep(0.08)


def send_chat_command(command: str) -> None:
    focus_toribash()
    try:
        p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
        p.communicate(command.encode("utf-8"))
        subprocess.run(["xdotool", "key", "t"], check=False)
        time.sleep(0.05)
        subprocess.run(["xdotool", "key", "ctrl+a"], check=False)
        time.sleep(0.02)
        subprocess.run(["xdotool", "key", "BackSpace"], check=False)
        time.sleep(0.02)
        subprocess.run(["xdotool", "key", "ctrl+v"], check=False)
        time.sleep(0.02)
        subprocess.run(["xdotool", "key", "Return"], check=False)
    except Exception:
        subprocess.run(["xdotool", "key", "t"], check=False)
        time.sleep(0.05)
        subprocess.run(["xdotool", "key", "ctrl+a"], check=False)
        subprocess.run(["xdotool", "key", "BackSpace"], check=False)
        subprocess.run(["xdotool", "type", "--delay", "1", command], check=False)
        subprocess.run(["xdotool", "key", "Return"], check=False)


def main() -> None:
    if not GENERATOR.exists():
        raise FileNotFoundError(GENERATOR)
    if not PROJECT_LUA.exists():
        raise FileNotFoundError(PROJECT_LUA)

    TORIBASH_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating sampled walking actions V25.1...")
    subprocess.run([str(ROOT / ".venv" / "bin" / "python3"), str(GENERATOR)], check=True)

    shutil.copy2(PROJECT_LUA, TORIBASH_LUA)
    print("Lua copied:", TORIBASH_LUA)
    print("Lua action table:", TORIBASH_SCRIPT_DIR / "curated_walking_sampled_v25_1_actions_table.lua")
    print()
    input("Entrée quand Toribash est ouvert... ")

    send_chat_command(RESET_COMMAND)
    time.sleep(0.6)
    send_chat_command(LUA_COMMAND)


if __name__ == "__main__":
    main()
