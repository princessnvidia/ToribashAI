-- toribash_xioi_master_final_v6.lua
-- Exact champion live runner.
-- Source action table: xioi_master_final_v6_exact_actions.lua
-- No mutation, no GRU, no pilot.

local ACTION_FILE = "xioi_master_final_v6_exact_actions.lua"

local actions = {}
local max_frame = 0
local action_count = 0
local pair_count = 0
local loaded = false
local load_error = "not loaded"

local tick = 0
local started = false
local auto_steps = 0
local last_action_frame = -1
local last_pairs = 0
local motor_ok = 0
local motor_fail = 0
local motor_method = "none"
local last_text = "none"

local function safe_dofile(path)
    local ok, err = pcall(function() dofile(path) end)
    return ok, err
end

local function load_actions()
    loaded = false
    load_error = "loading"

    local ok, err = safe_dofile(ACTION_FILE)
    if not ok then
        load_error = "dofile failed: " .. tostring(err)
        return
    end

    if type(XIOI_MASTER_FINAL_V6_ACTIONS) ~= "table" then
        load_error = "missing ACTIONS table"
        return
    end

    actions = XIOI_MASTER_FINAL_V6_ACTIONS
    max_frame = tonumber(XIOI_MASTER_FINAL_V6_MAX_FRAME or 0) or 0
    action_count = tonumber(XIOI_MASTER_FINAL_V6_ACTION_COUNT or 0) or 0
    pair_count = tonumber(XIOI_MASTER_FINAL_V6_PAIR_COUNT or 0) or 0
    loaded = true
    load_error = "ok"
end

local function apply_joint(j, v)
    local ok = false

    if set_joint_state then
        ok = pcall(function() set_joint_state(0, j, v) end)
        if ok then
            motor_method = "set_joint_state(0,j,v)"
            motor_ok = motor_ok + 1
            return true
        end

        ok = pcall(function() set_joint_state(j, v) end)
        if ok then
            motor_method = "set_joint_state(j,v)"
            motor_ok = motor_ok + 1
            return true
        end
    end

    motor_fail = motor_fail + 1
    return false
end

local function get_frame()
    if get_world_state then
        local ok, ws = pcall(function() return get_world_state() end)
        if ok and type(ws) == "table" then
            if ws.match_frame then return tonumber(ws.match_frame) or tick end
            if ws.game_frame then return tonumber(ws.game_frame) or tick end
            if ws.frame then return tonumber(ws.frame) or tick end
        end
    end
    return tick
end

local function auto_run_once()
    -- We use several possible APIs because Toribash Lua differs between builds.
    if run_frames then
        pcall(function() run_frames(1) end)
        auto_steps = auto_steps + 1
        return
    end
    if step_game then
        pcall(function() step_game() end)
        auto_steps = auto_steps + 1
        return
    end
    if run_cmd then
        pcall(function() run_cmd("/sp 1") end)
        auto_steps = auto_steps + 1
        return
    end
end

local function on_frame()
    if not loaded then return end

    local frame = get_frame()
    tick = tick + 1

    -- Some Toribash builds keep game paused until space once. This keeps pushing.
    if tick <= 8 or (tick % 2 == 0 and frame <= max_frame + 20) then
        auto_run_once()
    end

    local pairs = actions[frame]
    if pairs == nil then
        -- fallback: when frame counter is not exact, use nearest 5-frame tick
        local rounded = math.floor(frame / 5) * 5
        pairs = actions[rounded]
        if pairs ~= nil then frame = rounded end
    end

    if pairs ~= nil then
        last_action_frame = frame
        last_pairs = #pairs
        last_text = "f" .. tostring(frame) .. " pairs=" .. tostring(#pairs)
        for _, p in ipairs(pairs) do
            apply_joint(tonumber(p[1]), tonumber(p[2]))
        end
    else
        last_pairs = 0
    end
end

local function on_draw()
    if draw_text then
        draw_text("xioi_master_final_v6 EXACT", 40, 40, 1)
        draw_text("loaded=" .. tostring(loaded) .. " err=" .. tostring(load_error), 40, 60, 1)
        draw_text("actions=" .. tostring(action_count) .. " pairs=" .. tostring(pair_count) .. " max=" .. tostring(max_frame), 40, 80, 1)
        draw_text("tick=" .. tostring(tick) .. " last=" .. tostring(last_text), 40, 100, 1)
        draw_text("last_pairs=" .. tostring(last_pairs) .. " ok=" .. tostring(motor_ok) .. " fail=" .. tostring(motor_fail), 40, 120, 1)
        draw_text("method=" .. tostring(motor_method) .. " auto=" .. tostring(auto_steps), 40, 140, 1)
        draw_text("source=xioi_master_final_v5_champion.rpl", 40, 160, 1)
    end
end

load_actions()

if add_hook then
    add_hook("enter_frame", "xioi_master_final_v6_enter", on_frame)
    add_hook("draw2d", "xioi_master_final_v6_draw", on_draw)
end

if echo then
    echo("xioi_master_final_v6 loaded: " .. tostring(load_error) .. " actions=" .. tostring(action_count))
end
