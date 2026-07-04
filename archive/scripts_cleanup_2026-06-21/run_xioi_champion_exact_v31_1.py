#!/usr/bin/env python3
"""Déploie V31.1 exact runner dans Toribash."""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
SCRIPTS = ROOT / "scripts"
STEAM_SCRIPT_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"
)

GENERATOR = SCRIPTS / "generate_xioi_champion_exact_actions_v31_1.py"
PROJECT_LUA = SCRIPTS / "toribash_xioi_champion_exact_runner_v31_1.lua"
STEAM_LUA = STEAM_SCRIPT_DIR / "toribash_xioi_champion_exact_runner_v31_1.lua"
LUA_COMMAND = "/ls toribash_xioi_champion_exact_runner_v31_1.lua"


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
    if not GENERATOR.exists():
        raise FileNotFoundError(GENERATOR)
    if not PROJECT_LUA.exists():
        raise FileNotFoundError(PROJECT_LUA)

    py = ROOT / ".venv" / "bin" / "python3"
    if not py.exists():
        py = Path("python3")

    print("Generating exact action table...")
    subprocess.run([str(py), str(GENERATOR)], check=True)

    STEAM_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_LUA, STEAM_LUA)

    print("Lua copied:", STEAM_LUA)
    print("Command:", LUA_COMMAND)
    input("Entrée quand Toribash est ouvert... ")
    send_chat_command(LUA_COMMAND)
    print("Dans Toribash: appuie sur Espace une fois si la simulation est en pause.")


if __name__ == "__main__":
    main()
