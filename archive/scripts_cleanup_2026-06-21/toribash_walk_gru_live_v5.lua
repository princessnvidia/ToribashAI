-- toribash_walk_gru_live_v5.lua
-- Live walk runner based on the old upright/recovery pattern:
-- global Lua agent, automatic physics/space, run_frames(1), result JSON.

local HOOK = "toribash_walk_gru_live_v5"
local AGENT_FILE = "toribashai_agent_current.lua"
local RESULT_FILE = "../data/script/toribash_walk_gru_live_result_v5.json"

local CONFIG = {
    frames_per_action = 5,
    max_frames = 1200,
    warmup_frames = 0,
    fall_z = 4.6,
    min_head_z = 6.0,
    min_chest_z = 5.6,
    min_hip_z = 5.0,
    early_stop_after = 260,
}

local AGENT_JOINTS = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}

local running = false
local finished = false
local started_physics = false
local boot_ticks = 0
local frame = 0
local start_y = nil
local best_y = nil
local best_progress_frame = 0
local ok_count = 0
local fail_count = 0
local last_action_index = 0
local last_pairs = 0
local reason = "running"
local fell = false

local function q(s)
    s = tostring(s or "")
    s = s:gsub("\\", "\\\\"):gsub('"', '\\"')
    return '"' .. s .. '"'
end

local function write_file(path, text)
    local f = io.open(path, "w")
    if not f then return false end
    f:write(text)
    f:close()
    return true
end

local function load_agent()
    TORIBASHAI_AGENT = nil
    local ok, result = pcall(dofile, AGENT_FILE)
    if ok and result then
        TORIBASHAI_AGENT = result
    end
    if not TORIBASHAI_AGENT or not TORIBASHAI_AGENT.actions then
        echo("[Walk GRU V5] agent table missing: " .. AGENT_FILE)
        return false
    end

    CONFIG.frames_per_action = tonumber(TORIBASHAI_AGENT.frames_per_action or CONFIG.frames_per_action) or CONFIG.frames_per_action
    CONFIG.max_frames = tonumber(TORIBASHAI_AGENT.max_frames or CONFIG.max_frames) or CONFIG.max_frames
    CONFIG.warmup_frames = tonumber(TORIBASHAI_AGENT.warmup_frames or CONFIG.warmup_frames) or CONFIG.warmup_frames
    return true
end

local function safe_set_joint(j, v)
    local ok = false
    if set_joint_state then
        ok = pcall(function() set_joint_state(0, j, v) end)
        if not ok then ok = pcall(function() set_joint_state(j, v) end) end
    elseif set_joint then
        ok = pcall(function() set_joint(0, j, v) end)
        if not ok then ok = pcall(function() set_joint(j, v) end) end
    end
    if ok then ok_count = ok_count + 1 else fail_count = fail_count + 1 end
end

local function hold_all()
    for _, j in ipairs(AGENT_JOINTS) do
        safe_set_joint(j, 3)
    end
end

local function current_action_index()
    if not TORIBASHAI_AGENT then return 1 end
    local actions = TORIBASHAI_AGENT.actions or {}
    local n = #actions
    if n <= 0 then return 1 end

    local idx = math.floor((frame - CONFIG.warmup_frames) / CONFIG.frames_per_action) + 1
    if idx < 1 then idx = 1 end
    if idx <= n then return idx end

    -- After launch+seed, loop over the learned walking cycle instead of freezing.
    local loop_start = tonumber(TORIBASHAI_AGENT.loop_start_action or TORIBASHAI_AGENT.protected_actions or 1) or 1
    if loop_start < 1 then loop_start = 1 end
    if loop_start > n then loop_start = 1 end
    local loop_len = n - loop_start + 1
    if loop_len <= 0 then return n end
    return loop_start + ((idx - loop_start) % loop_len)
end

local function apply_agent_action()
    if not TORIBASHAI_AGENT or not TORIBASHAI_AGENT.actions then return end
    local idx = current_action_index()
    local action = TORIBASHAI_AGENT.actions[idx]
    if not action then return end

    last_action_index = idx
    last_pairs = 0
    local control_joints = TORIBASHAI_AGENT.control_joints or AGENT_JOINTS
    for i, joint_id in ipairs(control_joints) do
        local v = tonumber(action[i] or 0) or 0
        if v > 0 then
            safe_set_joint(joint_id, v)
            last_pairs = last_pairs + 1
        end
    end
end

local function body_pos(player, body)
    if get_body_info then
        local ok, info = pcall(function() return get_body_info(player, body) end)
        if ok and info then
            if info.pos then return info.pos.x or info.pos[1], info.pos.y or info.pos[2], info.pos.z or info.pos[3] end
            if info.x and info.y and info.z then return info.x, info.y, info.z end
            if info[1] and info[2] and info[3] then return info[1], info[2], info[3] end
        end
    end
    if get_body_position then
        local ok, x, y, z = pcall(function() return get_body_position(player, body) end)
        if ok and x then return x, y, z end
    end
    return nil, nil, nil
end

local function avg_body(ids)
    local sx, sy, sz, c = 0, 0, 0, 0
    for _, id in ipairs(ids) do
        local x, y, z = body_pos(0, id)
        if x and y and z then sx = sx + x; sy = sy + y; sz = sz + z; c = c + 1 end
    end
    if c == 0 then return nil end
    return sx / c, sy / c, sz / c
end

local function get_center()
    return avg_body({0,1,2,3,4,5})
end

local function posture()
    local hx, hy, hz = body_pos(0, 0)
    local cx, cy, cz = avg_body({1,2,3})
    local px, py, pz = avg_body({4,5,14,15})
    return hx, hy, hz, cx, cy, cz, px, py, pz
end

local function finish_run(r)
    if finished then return end
    finished = true
    running = false
    reason = r or reason

    local progress = 0
    if start_y and best_y then progress = start_y - best_y end -- our walk direction is negative Y
    local score = progress * 130.0 + frame * 0.30
    if fell then score = score - 350 end
    if reason == "stalled" then score = score - 120 end

    local txt = "{\n" ..
        '  "score": ' .. tostring(score) .. ",\n" ..
        '  "reason": ' .. q(reason) .. ",\n" ..
        '  "frames": ' .. tostring(frame) .. ",\n" ..
        '  "progress_y": ' .. tostring(progress) .. ",\n" ..
        '  "start_y": ' .. tostring(start_y or 0) .. ",\n" ..
        '  "best_y": ' .. tostring(best_y or 0) .. ",\n" ..
        '  "fell": ' .. tostring(fell) .. ",\n" ..
        '  "ok_count": ' .. tostring(ok_count) .. ",\n" ..
        '  "fail_count": ' .. tostring(fail_count) .. ",\n" ..
        '  "last_action_index": ' .. tostring(last_action_index) .. ",\n" ..
        '  "agent": ' .. q(TORIBASHAI_AGENT and TORIBASHAI_AGENT.name or "missing") .. ",\n" ..
        '  "generation": ' .. tostring(TORIBASHAI_AGENT and TORIBASHAI_AGENT.generation or -1) .. ",\n" ..
        '  "candidate": ' .. tostring(TORIBASHAI_AGENT and TORIBASHAI_AGENT.agent_index or -1) .. "\n" ..
        "}\n"
    write_file(RESULT_FILE, txt)
    echo("[Walk GRU V5] done score=" .. tostring(score) .. " reason=" .. tostring(reason) .. " progress=" .. tostring(progress))
    if freeze_game then freeze_game() end
end

local function start_physics_once()
    if started_physics then return end
    started_physics = true
    echo("[Walk GRU V5] START physics auto-space")
    if unfreeze_game then pcall(unfreeze_game) end
    if toggle_game_pause then pcall(function() toggle_game_pause(false) end) end
    if step_game then pcall(function() step_game(false, false) end) end
    if run_frames then
        pcall(function() run_frames(1) end)
        pcall(function() run_frames(10) end)
    end
end

local function reset_state()
    frame = 0
    boot_ticks = 0
    running = true
    finished = false
    started_physics = false
    fell = false
    reason = "running"
    start_y = nil
    best_y = nil
    best_progress_frame = 0
    ok_count = 0
    fail_count = 0
    last_action_index = 0
    last_pairs = 0
    hold_all()
end

local function on_new_game()
    local ok = load_agent()
    if not ok then
        running = false
        if freeze_game then freeze_game() end
        return
    end
    echo("[Walk GRU V5] loaded gen=" .. tostring(TORIBASHAI_AGENT.generation or 0) .. " agent=" .. tostring(TORIBASHAI_AGENT.agent_index or 0) .. " actions=" .. tostring(#TORIBASHAI_AGENT.actions))
    reset_state()
    if unfreeze_game then pcall(unfreeze_game) end
end

local function on_draw2d()
    if not running then return end
    if draw_text then
        draw_text("WalkGRU V5 gen=" .. tostring(TORIBASHAI_AGENT and TORIBASHAI_AGENT.generation or -1) ..
            " agent=" .. tostring(TORIBASHAI_AGENT and TORIBASHAI_AGENT.agent_index or -1), 40, 90, 1)
        draw_text("frame=" .. tostring(frame) .. " action=" .. tostring(last_action_index) .. " pairs=" .. tostring(last_pairs) ..
            " ok=" .. tostring(ok_count) .. " fail=" .. tostring(fail_count), 40, 120, 1)
    end
    if not started_physics then
        boot_ticks = boot_ticks + 1
        if boot_ticks >= 12 then start_physics_once() end
    end
end

local function on_enter_frame()
    if not running or finished then return end
    frame = frame + 1

    local x, y, z = get_center()
    if y and not start_y then
        start_y = y
        best_y = y
        best_progress_frame = frame
    end
    if y and (not best_y or y < best_y) then
        best_y = y
        best_progress_frame = frame
    end

    if frame < CONFIG.warmup_frames then
        hold_all()
    else
        apply_agent_action()
        apply_agent_action()
    end

    local hx, hy, hz, cx, cy, cz, px, py, pz = posture()
    if frame > 40 then
        if hz and hz < CONFIG.min_head_z then fell = true; return finish_run("head_low") end
        if cz and cz < CONFIG.min_chest_z then fell = true; return finish_run("chest_low") end
        if pz and pz < CONFIG.min_hip_z then fell = true; return finish_run("hip_low") end
    end

    if frame > 420 and frame - best_progress_frame > CONFIG.early_stop_after then
        return finish_run("stalled")
    end
    if frame >= CONFIG.max_frames then
        return finish_run("max_frames")
    end
    if z and z < CONFIG.fall_z then
        fell = true
        return finish_run("fell")
    end

    if run_frames then pcall(function() run_frames(1) end) end
end

remove_hooks(HOOK)
remove_hooks("toribash_walk_gru_live_v1")
remove_hooks("toribash_walk_gru_live_v2")
remove_hooks("toribash_walk_gru_live_v3")
remove_hooks("toribash_walk_gru_live_v4")
remove_hooks("toribashai_upright_runner_v18")
remove_hooks("toribashai_recovery_runner_v1")

add_hook("new_game", HOOK, on_new_game)
add_hook("draw2d", HOOK, on_draw2d)
add_hook("enter_frame", HOOK, on_enter_frame)

echo("[Walk GRU V5] loaded. Use /lm ToribashAI/toribashai_xioi_city_v1.tbm or reset from Python.")
