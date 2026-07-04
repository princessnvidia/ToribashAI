#!/usr/bin/env python3
"""
run_xioi_master_final_v6_1.py

Generates exact Lua action table from xioi_master_final_v5_champion.rpl,
copies the exact runner to Steam, and loads it in Toribash.
"""
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

GENERATOR = SCRIPTS / "generate_xioi_master_final_v6_1_exact_actions.py"
PROJECT_LUA = SCRIPTS / "toribash_xioi_master_final_v6_1.lua"
STEAM_LUA = STEAM_SCRIPT_DIR / "toribash_xioi_master_final_v6_1.lua"
LUA_COMMAND = "/ls toribash_xioi_master_final_v6_1.lua"


def focus_toribash() -> None:
    subprocess.run(["xdotool", "search", "--name", "Toribash", "windowactivate", "--sync"], check=False)
    time.sleep(0.06)


def send_chat_command(command: str) -> None:
    focus_toribash()
    try:
        p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
        p.communicate(command.encode("utf-8"))
        subprocess.run(["xdotool", "key", "t"], check=False)
        time.sleep(0.035)
        subprocess.run(["xdotool", "key", "ctrl+a"], check=False)
        time.sleep(0.015)
        subprocess.run(["xdotool", "key", "BackSpace"], check=False)
        time.sleep(0.015)
        subprocess.run(["xdotool", "key", "ctrl+v"], check=False)
        time.sleep(0.015)
        subprocess.run(["xdotool", "key", "Return"], check=False)
    except Exception:
        subprocess.run(["xdotool", "key", "t"], check=False)
        time.sleep(0.035)
        subprocess.run(["xdotool", "key", "ctrl+a"], check=False)
        subprocess.run(["xdotool", "key", "BackSpace"], check=False)
        subprocess.run(["xdotool", "type", "--delay", "1", command], check=False)
        subprocess.run(["xdotool", "key", "Return"], check=False)


def main() -> None:
    if not GENERATOR.exists():
        raise FileNotFoundError(GENERATOR)
    if not PROJECT_LUA.exists():
        raise FileNotFoundError(PROJECT_LUA)

    print("Generating exact actions from champion...")
    subprocess.run([str(ROOT / ".venv" / "bin" / "python3"), str(GENERATOR)], check=True)

    STEAM_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_LUA, STEAM_LUA)
    print("Lua copied:", STEAM_LUA)
    print("Command:", LUA_COMMAND)
    input("Entrée quand Toribash est ouvert... ")
    send_chat_command(LUA_COMMAND)
    print("Loaded. If needed, press Space once; it should then continue automatically.")


if __name__ == "__main__":
    main()
