#!/usr/bin/env python3
"""
run_xioi_champion_v31_live.py

Génère les actions V31, copie le Lua runner dans Toribash, puis charge le script.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
SCRIPTS = ROOT / "scripts"
TORIBASH_SCRIPT_DIR = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"

GENERATOR = SCRIPTS / "generate_xioi_champion_live_actions_v31.py"
PROJECT_LUA = SCRIPTS / "toribash_xioi_champion_live_runner_v31.lua"
TORIBASH_LUA = TORIBASH_SCRIPT_DIR / "toribash_xioi_champion_live_runner_v31.lua"
LUA_COMMAND = "/ls toribash_xioi_champion_live_runner_v31.lua"


def focus_toribash() -> None:
    subprocess.run(["xdotool", "search", "--name", "Toribash", "windowactivate", "--sync"], check=False)
    time.sleep(0.08)


def send_chat_command(command: str) -> None:
    focus_toribash()
    try:
        p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
        p.communicate(command.encode("utf-8"))
        subprocess.run(["xdotool", "key", "t"], check=False)
        time.sleep(0.04)
        subprocess.run(["xdotool", "key", "ctrl+a"], check=False)
        time.sleep(0.02)
        subprocess.run(["xdotool", "key", "BackSpace"], check=False)
        time.sleep(0.02)
        subprocess.run(["xdotool", "key", "ctrl+v"], check=False)
        time.sleep(0.02)
        subprocess.run(["xdotool", "key", "Return"], check=False)
    except Exception:
        subprocess.run(["xdotool", "key", "t"], check=False)
        subprocess.run(["xdotool", "type", "--delay", "1", command], check=False)
        subprocess.run(["xdotool", "key", "Return"], check=False)


def main() -> None:
    py = ROOT / ".venv" / "bin" / "python3"
    if not py.exists():
        py = Path("python3")

    print("Generating V31 actions...")
    subprocess.run([str(py), str(GENERATOR)], check=True)

    if not PROJECT_LUA.exists():
        raise FileNotFoundError(PROJECT_LUA)
    TORIBASH_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_LUA, TORIBASH_LUA)

    print("Lua copied:", TORIBASH_LUA)
    print("Action table should be in:", TORIBASH_SCRIPT_DIR / "xioi_champion_v31_live_actions_table.lua")
    input("Entrée quand Toribash est ouvert... ")
    send_chat_command(LUA_COMMAND)
    print("Loaded. Appuie une fois sur Espace si la simulation est en pause.")


if __name__ == "__main__":
    main()
