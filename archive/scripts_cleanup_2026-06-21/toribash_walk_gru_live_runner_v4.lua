-- toribash_walk_gru_live_runner_v4.lua
-- Walk GRU live runner V4
-- Reads toribash_walk_gru_live_agent_v4.lua
-- Applies a seeded walking cycle live, scores distance/upright time, auto-runs frames.

local AGENT_FILE = "../data/script/toribash_walk_gru_live_agent_v4.lua"
local RESULT_FILE = "../data/script/toribash_walk_gru_live_result_v4.json"

local agent = nil
local action_by_frame = {}
local loaded = false
local started = false
local last_action_frame = -1
local last_pairs = 0
local ok_count = 0
local fail_count = 0
local start_y = nil
local best_y = nil
local last_progress_frame = 0
local fell = false
local done = false
local reason = "running"
local frame_now = 0

local MAX_FRAMES = 1200
local EARLY_STOP_AFTER = 220
local MIN_HEAD_Z = 6.5
local MIN_CHEST_Z = 5.8
local MIN_HIP_Z = 4.8

local function file_read(path)
    local f = io.open(path, "r")
    if not f then return nil end
    local s = f:read("*all")
    f:close()
    return s
end

local function file_write(path, text)
    local f = io.open(path, "w")
    if not f then return false end
    f:write(text)
    f:close()
    return true
end

local function q(s)
    s = tostring(s or "")
    s = s:gsub("\\", "\\\\"):gsub('"', '\\"')
    return '"' .. s .. '"'
end

local function load_agent()
    local ok, result = pcall(dofile, AGENT_FILE)
    if ok then
        agent = result or TORIBASH_WALK_GRU_AGENT_V4 or TORIBASHAI_AGENT
    else
        agent = TORIBASH_WALK_GRU_AGENT_V4 or TORIBASHAI_AGENT
    end

    if not agent or not agent.actions then
        loaded = false
        return false
    end

    MAX_FRAMES = tonumber(agent.max_frames or MAX_FRAMES) or MAX_FRAMES
    EARLY_STOP_AFTER = tonumber(agent.early_stop_after or EARLY_STOP_AFTER) or EARLY_STOP_AFTER

    action_by_frame = {}
    for _, a in ipairs(agent.actions) do
        local fr = tonumber(a.frame or 0) or 0
        action_by_frame[fr] = a.pairs or {}
    end

    loaded = true
    return true
end

local function get_body_pos(part)
    if get_body_info then
        local ok, info = pcall(function() return get_body_info(0, part) end)
        if ok and info then
            if type(info) == "table" then
                local x = info.x or info.pos_x or info[1] or 0
                local y = info.y or info.pos_y or info[2] or 0
                local z = info.z or info.pos_z or info[3] or 0
                return x, y, z
            end
        end
    end
    if get_body_position then
        local ok, x, y, z = pcall(function() return get_body_position(0, part) end)
        if ok and x then return x, y, z end
    end
    return nil, nil, nil
end

local function safe_set_joint(j, v)
    local ok = false
    if set_joint_state then
        ok = pcall(function() set_joint_state(0, j, v) end)
        if not ok then
            ok = pcall(function() set_joint_state(j, v) end)
        end
    elseif set_joint then
        ok = pcall(function() set_joint(0, j, v) end)
        if not ok then ok = pcall(function() set_joint(j, v) end) end
    end
    if ok then ok_count = ok_count + 1 else fail_count = fail_count + 1 end
end

local function apply_action_for_frame(fr)
    local pairs = action_by_frame[fr]
    if not pairs then
        return
    end
    last_action_frame = fr
    last_pairs = #pairs
    for _, p in ipairs(pairs) do
        local j = tonumber(p[1])
        local v = tonumber(p[2])
        if j and v and v > 0 then
            safe_set_joint(j, v)
        end
    end
end

local function maybe_score()
    local hx, hy, hz = get_body_pos(0)      -- head-ish fallback if body ids align
    local cx, cy, cz = get_body_pos(2)      -- chest-ish
    local lx, ly, lz = get_body_pos(3)      -- lumbar/hips-ish

    local y = cy or ly or hy or 0
    local head_z = hz or 99
    local chest_z = cz or 99
    local hip_z = lz or 99

    if start_y == nil then
        start_y = y
        best_y = y
        last_progress_frame = frame_now
    end

    if best_y == nil or y < best_y then
        -- Xioi/assassin motion in our data goes toward negative Y.
        best_y = y
        last_progress_frame = frame_now
    end

    if head_z < MIN_HEAD_Z or chest_z < MIN_CHEST_Z or hip_z < MIN_HIP_Z then
        fell = true
        reason = "fell"
    end

    if frame_now >= MAX_FRAMES then
        reason = "max_frames"
        done = true
    end

    if frame_now > 350 and frame_now - last_progress_frame > EARLY_STOP_AFTER then
        reason = "stalled"
        done = true
    end

    if fell then done = true end

    if done then
        local progress = 0
        if start_y and best_y then progress = start_y - best_y end
        local score = progress * 100.0 + frame_now * 0.35
        if fell then score = score - 250 end
        if reason == "stalled" then score = score - 80 end
        local txt = "{\n" ..
            '  "score": ' .. tostring(score) .. ",\n" ..
            '  "reason": ' .. q(reason) .. ",\n" ..
            '  "frames": ' .. tostring(frame_now) .. ",\n" ..
            '  "progress_y": ' .. tostring(progress) .. ",\n" ..
            '  "best_y": ' .. tostring(best_y or 0) .. ",\n" ..
            '  "start_y": ' .. tostring(start_y or 0) .. ",\n" ..
            '  "fell": ' .. tostring(fell) .. ",\n" ..
            '  "last_action_frame": ' .. tostring(last_action_frame) .. ",\n" ..
            '  "ok_count": ' .. tostring(ok_count) .. ",\n" ..
            '  "fail_count": ' .. tostring(fail_count) .. ",\n" ..
            '  "agent": ' .. q(agent and agent.name or "missing") .. ",\n" ..
            '  "generation": ' .. tostring(agent and agent.generation or -1) .. ",\n" ..
            '  "candidate": ' .. tostring(agent and agent.candidate or -1) .. "\n" ..
            "}\n"
        file_write(RESULT_FILE, txt)
    end
end

local function boot_game()
    if not loaded then load_agent() end
    started = true
    done = false
    fell = false
    reason = "running"
    ok_count = 0
    fail_count = 0
    frame_now = 0
    last_action_frame = -1
    last_pairs = 0
    start_y = nil
    best_y = nil
    last_progress_frame = 0
end

local function on_enter_frame()
    if done then return end
    if not loaded then load_agent() end
    if not started then boot_game() end

    if get_world_state then
        local ok, ws = pcall(function() return get_world_state() end)
        if ok and ws and ws.match_frame then
            frame_now = tonumber(ws.match_frame) or frame_now
        else
            frame_now = frame_now + 1
        end
    else
        frame_now = frame_now + 1
    end

    if frame_now % 5 == 0 then
        apply_action_for_frame(frame_now)
    end
    maybe_score()

    if run_frames and not done then
        pcall(function() run_frames(1) end)
    end
end

local function draw_overlay()
    set_color(1, 1, 1, 1)
    local g = agent and tostring(agent.generation or "?") or "?"
    local c = agent and tostring(agent.candidate or "?") or "?"
    local nm = agent and tostring(agent.name or "agent") or "agent?"
    draw_text("walk_gru_live_v4 gen=" .. g .. " agent=" .. c, 40, 80, 1)
    draw_text(nm, 40, 105, 1)
    draw_text("frame=" .. tostring(frame_now) .. " last_action=" .. tostring(last_action_frame) .. " pairs=" .. tostring(last_pairs), 40, 130, 1)
    draw_text("ok=" .. tostring(ok_count) .. " fail=" .. tostring(fail_count) .. " reason=" .. tostring(reason), 40, 155, 1)
end

load_agent()
add_hook("enter_frame", "walk_gru_live_v4_enter", on_enter_frame)
add_hook("draw2d", "walk_gru_live_v4_draw", draw_overlay)
echo("walk_gru_live_v4 loaded")
