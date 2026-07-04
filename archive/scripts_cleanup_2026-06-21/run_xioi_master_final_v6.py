#!/usr/bin/env python3
"""
run_xioi_master_final_v6.py

Deploys exact champion runner V6 to Toribash.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
SCRIPTS = ROOT / "scripts"
GENERATOR = SCRIPTS / "generate_xioi_master_final_v6_exact_actions.py"
PROJECT_LUA = SCRIPTS / "toribash_xioi_master_final_v6.lua"

STEAM_SCRIPT_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"
)
STEAM_LUA = STEAM_SCRIPT_DIR / "toribash_xioi_master_final_v6.lua"

LUA_COMMAND = "/ls toribash_xioi_master_final_v6.lua"
RESET_COMMAND = "/reset"


def focus_toribash():
    subprocess.run(["xdotool", "search", "--name", "Toribash", "windowactivate", "--sync"], check=False)
    time.sleep(0.05)


def send_chat_command(command: str):
    focus_toribash()
    try:
        p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
        p.communicate(command.encode("utf-8"))
        subprocess.run(["xdotool", "key", "t"], check=False)
        time.sleep(0.03)
        subprocess.run(["xdotool", "key", "ctrl+a"], check=False)
        time.sleep(0.01)
        subprocess.run(["xdotool", "key", "BackSpace"], check=False)
        time.sleep(0.01)
        subprocess.run(["xdotool", "key", "ctrl+v"], check=False)
        time.sleep(0.01)
        subprocess.run(["xdotool", "key", "Return"], check=False)
    except Exception:
        subprocess.run(["xdotool", "key", "t"], check=False)
        subprocess.run(["xdotool", "type", "--delay", "1", command], check=False)
        subprocess.run(["xdotool", "key", "Return"], check=False)


def main():
    if not GENERATOR.exists():
        raise FileNotFoundError(GENERATOR)
    if not PROJECT_LUA.exists():
        raise FileNotFoundError(PROJECT_LUA)

    print("Generating exact action table...")
    subprocess.run([str(ROOT / ".venv" / "bin" / "python3"), str(GENERATOR)], check=True)

    STEAM_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_LUA, STEAM_LUA)
    print("Lua copied:", STEAM_LUA)

    input("Entrée quand Toribash est ouvert... ")
    send_chat_command(RESET_COMMAND)
    time.sleep(0.25)
    send_chat_command(LUA_COMMAND)
    time.sleep(0.15)
    # fast first space after loading, like older working runners
    focus_toribash()
    subprocess.run(["xdotool", "key", "space"], check=False)
    print("Loaded V6 exact runner. If needed, press Space once in Toribash.")


if __name__ == "__main__":
    main()
