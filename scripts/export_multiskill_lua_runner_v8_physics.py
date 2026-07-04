#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
STEAM_TB = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
DATA = ROOT / "evolution" / "multiskill_actions_v8_physics.json"
OUT = STEAM_TB / "data" / "script" / "toribash_multiskill_runner_v8.lua"

data = json.loads(DATA.read_text())

launch = [a["joints"] for a in data["skills"]["launch"]["actions"]]
walk = [a["joints"] for a in data["skills"]["walk"]["actions"]]

def lua_table(actions):
    return "{\n" + "\n".join(
        "  {" + ", ".join(map(str, a)) + "}," for a in actions
    ) + "\n}"

lua = f'''-- ToribashAI Multiskill Runner V8 Physics
echo("[ToribashAI V8] loaded")

local launch_actions = {lua_table(launch)}
local walk_actions = {lua_table(walk)}

local phase = "launch"
local i = 1
local frame = 0
local frames_per_action = 5
local running = true

local function apply_action(a)
    for j = 0, 19 do
        set_joint_state(0, j, a[j + 1], true)
    end
end

local function reset_ai()
    phase = "launch"
    i = 1
    frame = 0
    running = true
    unfreeze_game()
    toggle_game_pause(false)
    echo("[ToribashAI V8] reset launch -> walk")
end

local function next_action()
    if phase == "launch" then
        local a = launch_actions[i]
        i = i + 1
        if i > #launch_actions then
            phase = "walk"
            i = 1
            echo("[ToribashAI V8] switch to walk")
        end
        return a
    end

    local a = walk_actions[i]
    i = i + 1
    if i > #walk_actions then
        i = 1
    end
    return a
end

add_hook("new_game", "toribashai_v8_newgame", reset_ai)

add_hook("enter_frame", "toribashai_v8_enterframe", function()
    if not running then return end
    frame = frame + 1

    if frame % frames_per_action == 0 then
        local a = next_action()
        if a then apply_action(a) end
    end

    run_frames(1)
end)

echo("[ToribashAI V8] launch actions: " .. #launch_actions)
echo("[ToribashAI V8] walk actions: " .. #walk_actions)
'''

OUT.write_text(lua, encoding="utf-8")
print(f"[OK] Lua écrit: {OUT}")
print(f"launch={len(launch)} walk={len(walk)}")
