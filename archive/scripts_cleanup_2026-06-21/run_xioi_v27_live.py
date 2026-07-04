#!/usr/bin/env python3
"""
run_xioi_v27_live.py

Déploie V27 teacher-forced / sampled table + runner Lua.
Par défaut génère la table teacher-forced depuis le modèle overfit si disponible.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
SCRIPTS = ROOT / "scripts"
TORIBASH_SCRIPT_DIR = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"

GENERATOR = SCRIPTS / "generate_xioi_teacher_forced_actions_v27.py"
PROJECT_LUA = SCRIPTS / "toribash_xioi_gru_live_runner_v27.lua"
TORIBASH_LUA = TORIBASH_SCRIPT_DIR / "toribash_xioi_gru_live_runner_v27.lua"
LUA_COMMAND = "/ls toribash_xioi_gru_live_runner_v27.lua"


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
        subprocess.run(["xdotool", "type", "--delay", "1", command], check=False)
        subprocess.run(["xdotool", "key", "Return"], check=False)


def main() -> None:
    if not GENERATOR.exists():
        raise FileNotFoundError(GENERATOR)
    if not PROJECT_LUA.exists():
        raise FileNotFoundError(PROJECT_LUA)

    print("Generating V27 teacher-forced action table...")
    subprocess.run([str(ROOT / ".venv" / "bin" / "python3"), str(GENERATOR)], check=True)

    TORIBASH_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT_LUA, TORIBASH_LUA)
    print("Lua copied:", TORIBASH_LUA)
    print("Command:", LUA_COMMAND)
    input("Entrée quand Toribash est ouvert... ")
    send_chat_command(LUA_COMMAND)


if __name__ == "__main__":
    main()
