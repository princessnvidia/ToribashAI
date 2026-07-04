-- toribash_curated_walking_sampled_runner_v25_1.lua
-- V25.1: runner live sans parser JSON. Charge une table Lua générée par Python.

local HOOK = "curated_walking_sampled_v25_1"
local ACTION_TABLE_FILE = "curated_walking_sampled_v25_1_actions_table.lua"

local action_by_frame = {}
local action_frames = {}
local loaded = false
local running = false
local finished = false

local frame = 0
local warmup_frames = 18
local max_frames = 950
local motor_ok = 0
local motor_fail = 0
local last_pairs = 0
local last_action_text = "none"
local last_motor_method = "none"
local status = "boot"
local auto_steps = 0

local function log(msg)
    pcall(function() echo("[" .. HOOK .. "] " .. tostring(msg)) end)
end

local function safe_dofile(path)
    local ok, result = pcall(function() return dofile(path) end)
    if ok then return result end
    status = "dofile_failed"
    log("dofile failed: " .. tostring(path) .. " err=" .. tostring(result))
    return nil
end

local function count_table(t)
    local n = 0
    if type(t) ~= "table" then return 0 end
    for _k, _v in pairs(t) do n = n + 1 end
    return n
end

local function load_actions()
    -- clear globals before loading, to avoid stale data
    ACTIONS_BY_FRAME = nil
    ACTION_FRAMES = nil
    CURATED_WALKING_V25_1_META = nil

    local result = safe_dofile(ACTION_TABLE_FILE)
    if type(result) == "table" then
        action_by_frame = result.actions or ACTIONS_BY_FRAME or {}
        action_frames = result.frames or ACTION_FRAMES or {}
    else
        action_by_frame = ACTIONS_BY_FRAME or {}
        action_frames = ACTION_FRAMES or {}
    end

    table.sort(action_frames)

    local n = #action_frames
    if n <= 0 then
        -- fallback for non-array frames
        action_frames = {}
        for fr, _pairs in pairs(action_by_frame) do
            table.insert(action_frames, tonumber(fr) or 0)
        end
        table.sort(action_frames)
        n = #action_frames
    end

    if n > 0 and count_table(action_by_frame) > 0 then
        loaded = true
        status = "loaded"
        log("actions loaded=" .. tostring(n) .. " table_entries=" .. tostring(count_table(action_by_frame)))
        return true
    end

    loaded = false
    status = "no_actions_loaded"
    log("no actions loaded from lua table")
    return false
end

local function apply_joint(j, v)
    local ok = false
    if set_joint_state then
        ok = pcall(function() set_joint_state(0, j, v) end)
        if ok then
            last_motor_method = "set_joint_state(0,j,v)"
        else
            ok = pcall(function() set_joint_state(j, v) end)
            if ok then last_motor_method = "set_joint_state(j,v)" end
        end
    end

    if ok then
        motor_ok = motor_ok + 1
    else
        motor_fail = motor_fail + 1
        last_motor_method = "fail"
    end
end

local function apply_due_actions()
    if frame < warmup_frames then
        last_pairs = 0
        return
    end

    local logical_frame = frame - warmup_frames
    local action_frame = logical_frame - (logical_frame % 5)
    local pairs_for_frame = action_by_frame[action_frame]

    if pairs_for_frame then
        last_pairs = #pairs_for_frame
        last_action_text = "f" .. tostring(action_frame) .. " pairs=" .. tostring(#pairs_for_frame)
        for _, p in ipairs(pairs_for_frame) do
            local j = tonumber(p[1])
            local v = tonumber(p[2])
            if j and v then apply_joint(j, v) end
        end
    else
        last_pairs = 0
    end
end

local function start_physics()
    pcall(function() unfreeze_game() end)
    pcall(function() toggle_game_pause(false) end)
    pcall(function() run_frames(1) end)
end

local function reset_state()
    frame = 0
    motor_ok = 0
    motor_fail = 0
    last_pairs = 0
    last_action_text = "none"
    last_motor_method = "none"
    auto_steps = 0
    finished = false
end

local function on_new_game()
    reset_state()
    if not load_actions() then
        running = false
        return
    end
    running = true
    status = "running"
    start_physics()
end

local function on_enter_frame()
    if not running or finished then return end

    apply_due_actions()
    frame = frame + 1

    if frame >= max_frames then
        finished = true
        running = false
        status = "finished"
        return
    end

    local ok = pcall(function() run_frames(1) end)
    if ok then auto_steps = auto_steps + 1 end
end

local function draw_overlay()
    pcall(function()
        set_color(1, 1, 1, 1)
        draw_text("Curated Walking SAMPLED V25.1 LUA TABLE", 40, 80, 1)
        draw_text("status=" .. tostring(status) .. " actions=" .. tostring(#action_frames) .. " entries=" .. tostring(count_table(action_by_frame)), 40, 100, 1)
        draw_text("frame=" .. tostring(frame) .. " pairs=" .. tostring(last_pairs) .. " auto=" .. tostring(auto_steps), 40, 120, 1)
        draw_text("last=" .. tostring(last_action_text), 40, 140, 1)
        draw_text("ok=" .. tostring(motor_ok) .. " fail=" .. tostring(motor_fail) .. " method=" .. tostring(last_motor_method), 40, 160, 1)
        draw_text("table=" .. ACTION_TABLE_FILE, 40, 180, 1)
    end)
end

pcall(function() remove_hooks("curated_walking_sampled_v25_1") end)
pcall(function() remove_hooks("curated_walking_sampled_v25") end)
pcall(function() remove_hooks("curated_walking_gru_v24_3") end)
pcall(function() remove_hooks("curated_walking_gru_v24_2") end)
pcall(function() remove_hooks("curated_walking_gru_v24") end)

add_hook("new_game", HOOK, on_new_game)
add_hook("enter_frame", HOOK, on_enter_frame)
add_hook("draw2d", HOOK, draw_overlay)

log("Lua loaded")
on_new_game()
