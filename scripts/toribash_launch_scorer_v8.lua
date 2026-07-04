-- ToribashAI Launch Scorer V8

local ACTIONS_PATH = "../data/script/toribashai_launch_actions_v8.lua"
local RESULT_PATH = "../data/script/toribashai_launch_result_v8.json"

local actions = {}
local step_i = 1
local frame_i = 0
local max_frames = 220
local start_y = nil
local best_y = -9999
local fallen = false

dofile(ACTIONS_PATH)

local function apply_action(action)
    for joint = 0, 19 do
        local state = action[joint + 1]
        if state ~= nil then
            set_joint_state(0, joint, state)
        end
    end
end

local function vec_y(v)
    if type(v) == "table" then return v.y or v[2] or 0 end
    return 0
end

local function vec_z(v)
    if type(v) == "table" then return v.z or v[3] or 0 end
    return 0
end

local function get_body(part)
    local ok, pos = pcall(get_body_info, 0, part)
    if ok and pos then return pos end
    return {x=0,y=0,z=0}
end

local function write_result(reason)
    local chest = get_body(1)
    local head = get_body(0)
    local hip = get_body(3)

    local y = vec_y(chest)
    local progress = y - (start_y or y)
    local head_z = vec_z(head)
    local hip_z = vec_z(hip)

    local score = progress * 100.0
    score = score + math.max(0, head_z - 4.0) * 5.0
    score = score + math.max(0, hip_z - 3.0) * 5.0

    if progress < -1.0 then score = score - 200 end
    if head_z < 2.0 then score = score - 300 end
    if hip_z < 1.6 then score = score - 250 end
    if fallen then score = score - 500 end

    local f = io.open(RESULT_PATH, "w")
    if f then
        f:write(string.format(
            '{"score": %.6f, "reason": "%s", "frames": %d, "progress_y": %.6f, "head_z": %.6f, "hip_z": %.6f}',
            score, reason, frame_i, progress, head_z, hip_z
        ))
        f:close()
    end
end

local function reset_runner()
    step_i = 1
    frame_i = 0
    start_y = nil
    best_y = -9999
    fallen = false
    echo("Launch scorer V8 reset")
end

add_hook("new_game", "toribashai_launch_v8_new_game", function()
    reset_runner()
end)

add_hook("enter_frame", "toribashai_launch_v8_enter_frame", function()
    frame_i = frame_i + 1

    local chest = get_body(1)
    local head = get_body(0)
    local hip = get_body(3)

    local y = vec_y(chest)
    if start_y == nil then start_y = y end
    if y > best_y then best_y = y end

    if vec_z(head) < 1.5 or vec_z(hip) < 1.2 then
        fallen = true
    end

    if step_i <= #launch_actions then
        apply_action(launch_actions[step_i])
        step_i = step_i + 1
    end

    if frame_i >= max_frames then
        write_result("max_frames")
    end
end)

echo("ToribashAI Launch Scorer V8 loaded")
