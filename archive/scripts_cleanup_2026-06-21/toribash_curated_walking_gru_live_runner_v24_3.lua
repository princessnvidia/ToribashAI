-- toribash_curated_walking_gru_live_runner_v24_3.lua
-- V24.3: reprend le comportement des anciens Lua: auto-advance avec run_frames(1).
-- Objectif: ne plus devoir appuyer plusieurs fois sur Espace.

local HOOK = "curated_walking_gru_v24_3"
local INPUT_FILE = "curated_walking_gru_v24_live_actions_current.json"

local actions = {}
local action_by_frame = {}
local sorted_frames = {}
local loaded = false
local running = false
local finished = false

local frame = 0
local warmup_frames = 18
local max_frames = 900
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

local function safe_read_file(path)
    local f = io.open(path, "r")
    if not f then
        status = "missing_actions_file"
        log("missing actions file: " .. tostring(path))
        return nil
    end

    local ok, s = pcall(function()
        return f:read("*all")
    end)
    f:close()

    if not ok or not s then
        status = "read_failed"
        log("read failed: " .. tostring(path))
        return nil
    end
    return s
end

local function parse_actions_json(txt)
    local parsed = {}
    if not txt or txt == "" then return parsed end

    -- Parse simple des blocs {"frame": N, "pairs": [[j,v], ...]}
    for block in txt:gmatch('%b{}') do
        local fr = block:match('"frame"%s*:%s*(%d+)')
        local pairs_part = block:match('"pairs"%s*:%s*(%b[])')
        if fr and pairs_part then
            local pairs = {}
            for j, v in pairs_part:gmatch('%[%s*(%d+)%s*,%s*(%d+)%s*%]') do
                table.insert(pairs, { tonumber(j), tonumber(v) })
            end
            table.insert(parsed, { frame = tonumber(fr), pairs = pairs })
        end
    end
    return parsed
end

local function load_actions()
    local txt = safe_read_file(INPUT_FILE)
    actions = parse_actions_json(txt)
    action_by_frame = {}
    sorted_frames = {}

    for _, a in ipairs(actions) do
        local fr = tonumber(a.frame) or 0
        action_by_frame[fr] = a.pairs or {}
        table.insert(sorted_frames, fr)
    end
    table.sort(sorted_frames)

    if #actions > 0 then
        loaded = true
        status = "loaded"
        log("actions loaded=" .. tostring(#actions))
        return true
    end

    loaded = false
    if status == "boot" then status = "no_actions_parsed" end
    log("no actions parsed")
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
    local pairs = action_by_frame[action_frame]

    if pairs then
        last_pairs = #pairs
        last_action_text = "f" .. tostring(action_frame) .. " pairs=" .. tostring(#pairs)
        for _, p in ipairs(pairs) do
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

    -- Comportement repris des anciens runners: on avance la physique automatiquement.
    local ok = pcall(function() run_frames(1) end)
    if ok then
        auto_steps = auto_steps + 1
    end
end

local function draw_overlay()
    pcall(function()
        set_color(1, 1, 1, 1)
        draw_text("Curated Walking GRU V24.3 AUTO", 40, 80, 1)
        draw_text("status=" .. tostring(status) .. " actions=" .. tostring(#actions), 40, 100, 1)
        draw_text("frame=" .. tostring(frame) .. " pairs=" .. tostring(last_pairs) .. " auto=" .. tostring(auto_steps), 40, 120, 1)
        draw_text("last=" .. tostring(last_action_text), 40, 140, 1)
        draw_text("ok=" .. tostring(motor_ok) .. " fail=" .. tostring(motor_fail) .. " method=" .. tostring(last_motor_method), 40, 160, 1)
        draw_text("file=" .. INPUT_FILE, 40, 180, 1)
    end)
end

pcall(function() remove_hooks("curated_walking_gru_v24_3") end)
pcall(function() remove_hooks("curated_walking_gru_v24_2") end)
pcall(function() remove_hooks("curated_walking_gru_v24") end)
pcall(function() remove_hooks("toribashai_curated_walking_gru_v24_2_step") end)
pcall(function() remove_hooks("toribashai_curated_walking_gru_v24_2_draw") end)

add_hook("new_game", HOOK, on_new_game)
add_hook("enter_frame", HOOK, on_enter_frame)
add_hook("draw2d", HOOK, draw_overlay)

log("Lua loaded")
on_new_game()
