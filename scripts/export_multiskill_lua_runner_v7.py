#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
STEAM_TB = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
DATA = ROOT / "evolution" / "multiskill_actions_v7.json"

OUT = STEAM_TB / "data" / "script" / "toribash_multiskill_runner_v7.lua"

data = json.loads(DATA.read_text())

launch = [a["joints"] for a in data["skills"]["launch"]["actions"]]
walk = [a["joints"] for a in data["skills"]["walk"]["actions"]]

def lua_table(actions):
    lines = ["{"]
    for arr in actions:
        lines.append("  {" + ", ".join(str(x) for x in arr) + "},")
    lines.append("}")
    return "\n".join(lines)

lua = f'''-- ToribashAI Multiskill Runner V7
-- launch once -> walk loop
-- generated automatically

local MOD_NAME = "ToribashAI/toribashai_xioi_city_v1.tbm"

local launch_actions = {lua_table(launch)}

local walk_actions = {lua_table(walk)}

local frame_count = 0
local phase = "launch"
local launch_i = 1
local walk_i = 1

local function apply_action(action)
    for joint = 0, 19 do
        local state = action[joint + 1]
        if state ~= nil then
            set_joint_state(0, joint, state)
        end
    end
end

local function current_action()
    if phase == "launch" then
        local action = launch_actions[launch_i]
        launch_i = launch_i + 1

        if launch_i > #launch_actions then
            phase = "walk"
            walk_i = 1
        end

        return action
    end

    local action = walk_actions[walk_i]
    walk_i = walk_i + 1

    if walk_i > #walk_actions then
        walk_i = 1
    end

    return action
end

local function step()
    local action = current_action()
    if action then
        apply_action(action)
    end

    frame_count = frame_count + 1
end

local function reset_ai()
    frame_count = 0
    phase = "launch"
    launch_i = 1
    walk_i = 1
    echo("ToribashAI V7 reset: launch -> walk loop")
end

add_hook("new_game", "toribashai_v7_new_game", function()
    reset_ai()
end)

add_hook("enter_frame", "toribashai_v7_enter_frame", function()
    step()
end)

echo("ToribashAI Multiskill Runner V7 loaded")
echo("Mod attendu: " .. MOD_NAME)
echo("launch actions: " .. #launch_actions)
echo("walk actions: " .. #walk_actions)
'''

OUT.write_text(lua, encoding="utf-8")
print(f"[OK] Lua écrit: {OUT}")
print(f"launch actions: {len(launch)}")
print(f"walk actions: {len(walk)}")
