-- toribash_xioi_champion_exact_runner_v31_1.lua
-- V31.1 exact runner: rejoue les JOINT exacts du champion V30/V31.

local ACTIONS_FILE = "xioi_champion_exact_actions_v31_1.lua"

local actions = {}
local loaded = false
local status = "boot"
local action_count = 0
local min_frame = 0
local max_frame = 0
local last_frame = -1
local last_pairs = 0
local ok_count = 0
local fail_count = 0
local method = "none"
local auto_steps = 0

local function safe_dofile(path)
    if dofile then
        local ok, data = pcall(dofile, path)
        if ok and data then return data end
        status = "dofile fail: " .. tostring(data)
        return nil
    end
    status = "no dofile"
    return nil
end

local function load_actions()
    local data = safe_dofile(ACTIONS_FILE)
    if not data or not data.actions then
        loaded = false
        status = "missing actions table"
        return
    end

    actions = data.actions
    loaded = true
    status = "loaded " .. tostring(data.name or "actions")
    action_count = tonumber(data.action_count or 0) or 0

    local first = nil
    local last = nil
    for fr, _ in pairs(actions) do
        local n = tonumber(fr)
        if n then
            if first == nil or n < first then first = n end
            if last == nil or n > last then last = n end
        end
    end
    min_frame = first or 0
    max_frame = last or 0
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

local function get_frame()
    if get_world_state then
        local ok, ws = pcall(get_world_state)
        if ok and ws and ws.match_frame then return tonumber(ws.match_frame) or 0 end
    end
    if get_game_time then
        local ok, t = pcall(get_game_time)
        if ok and t then return tonumber(t) or 0 end
    end
    return 0
end

local function step_auto()
    -- Reprend le comportement des anciens runners: une fois lancé par espace,
    -- on avance automatiquement frame par frame si l'API existe.
    if run_frames then
        pcall(function() run_frames(1) end)
        auto_steps = auto_steps + 1
    elseif step_game then
        pcall(function() step_game() end)
        auto_steps = auto_steps + 1
    end
end

local function on_enter_frame()
    if not loaded then return end

    local fr = get_frame()
    local pairs = actions[fr]

    if pairs ~= nil then
        last_frame = fr
        last_pairs = #pairs
        for _, p in ipairs(pairs) do
            local j = tonumber(p[1])
            local v = tonumber(p[2])
            if j and v then apply_joint(j, v) end
        end
    else
        last_pairs = 0
    end

    if fr <= max_frame + 10 then
        step_auto()
    end
end

local function draw_overlay()
    if draw_text then
        set_color(1, 1, 1, 1)
        draw_text("Xioi Champion Exact V31.1", 40, 40, 1)
        draw_text("status=" .. tostring(status), 40, 60, 1)
        draw_text("actions=" .. tostring(action_count) .. " range=" .. tostring(min_frame) .. "-" .. tostring(max_frame), 40, 80, 1)
        draw_text("last_frame=" .. tostring(last_frame) .. " pairs=" .. tostring(last_pairs), 40, 100, 1)
        draw_text("ok=" .. tostring(ok_count) .. " fail=" .. tostring(fail_count), 40, 120, 1)
        draw_text("method=" .. tostring(method), 40, 140, 1)
        draw_text("auto=" .. tostring(auto_steps) .. " file=" .. ACTIONS_FILE, 40, 160, 1)
    end
end

load_actions()

add_hook("enter_frame", "xioi_champion_exact_v31_1_enter", on_enter_frame)
add_hook("draw2d", "xioi_champion_exact_v31_1_draw", draw_overlay)
