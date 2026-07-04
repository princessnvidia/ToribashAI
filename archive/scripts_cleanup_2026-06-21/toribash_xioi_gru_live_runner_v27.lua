-- toribash_xioi_gru_live_runner_v27.lua
-- V27 runner: lit une table Lua native xioi_v27_live_actions_table.lua
-- Un seul espace au début suffit normalement; ensuite auto-run.

local TABLE_FILE = "xioi_v27_live_actions_table.lua"
local ACTIONS = {}
local ACTION_COUNT = 0
local TURNFRAMES = 5
local loaded = false
local status = "init"
local frame_tick = 0
local last_pairs = 0
local last_action_frame = -1
local ok_count = 0
local fail_count = 0
local method = "none"
local auto_steps = 0

local function safe_dofile(path)
    local ok, err = pcall(function() dofile(path) end)
    if not ok then return false, tostring(err) end
    return true, "ok"
end

local function load_actions()
    XIOI_V27_ACTIONS = nil
    XIOI_V27_ACTION_COUNT = nil
    XIOI_V27_TURNFRAMES = nil

    local ok, err = safe_dofile(TABLE_FILE)
    if not ok then
        loaded = false
        status = "missing table: " .. err
        echo("[xioi_v27] " .. status)
        return
    end

    ACTIONS = XIOI_V27_ACTIONS or {}
    ACTION_COUNT = XIOI_V27_ACTION_COUNT or 0
    TURNFRAMES = XIOI_V27_TURNFRAMES or 5
    loaded = true
    status = "loaded"
    echo("[xioi_v27] actions loaded=" .. tostring(ACTION_COUNT))
end

local function apply_joint(j, v)
    local ok = false
    if set_joint_state then
        ok = pcall(function() set_joint_state(0, j, v) end)
        if ok then
            method = "set_joint_state(0,j,v)"
        else
            ok = pcall(function() set_joint_state(j, v) end)
            if ok then method = "set_joint_state(j,v)" end
        end
    end
    if ok then ok_count = ok_count + 1 else fail_count = fail_count + 1 end
end

local function apply_current_action()
    if not loaded then return end
    local action_frame = math.floor(frame_tick / TURNFRAMES) * TURNFRAMES
    local pairs = ACTIONS[action_frame]
    if pairs == nil then
        last_pairs = 0
        return
    end
    last_action_frame = action_frame
    last_pairs = #pairs
    for _, p in ipairs(pairs) do
        if p and p[1] ~= nil and p[2] ~= nil then
            apply_joint(tonumber(p[1]), tonumber(p[2]))
        end
    end
end

local function auto_run_one()
    if run_frames then
        pcall(function() run_frames(1) end)
    elseif step_game then
        pcall(function() step_game() end)
    end
end

local function on_enter_frame()
    if not loaded then return end
    apply_current_action()
    frame_tick = frame_tick + 1
    auto_steps = auto_steps + 1
    if auto_steps < 1600 then auto_run_one() end
end

local function on_draw2d()
    set_color(1, 1, 1, 1)
    draw_text("Xioi V27 teacher/sample/evo", 40, 40, 1)
    draw_text("status=" .. tostring(status) .. " actions=" .. tostring(ACTION_COUNT), 40, 60, 1)
    draw_text("tick=" .. tostring(frame_tick) .. " action_f=" .. tostring(last_action_frame) .. " pairs=" .. tostring(last_pairs), 40, 80, 1)
    draw_text("ok=" .. tostring(ok_count) .. " fail=" .. tostring(fail_count) .. " method=" .. tostring(method), 40, 100, 1)
    draw_text("table=" .. TABLE_FILE, 40, 120, 1)
end

load_actions()
add_hook("enter_frame", "xioi_v27_enter_frame", on_enter_frame)
add_hook("draw2d", "xioi_v27_draw2d", on_draw2d)
