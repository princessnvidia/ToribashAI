-- toribash_xioi_champion_live_runner_v31.lua
-- Runner live pour table Lua xioi_champion_v31_live_actions_table.lua

local TABLE_FILE = "xioi_champion_v31_live_actions_table.lua"
local TURNFRAMES = 5
local tick = 0
local loaded = false
local load_error = ""
local actions = {}
local action_count = 0
local motor_ok = 0
local motor_fail = 0
local last_pairs = 0
local last_frame = -1
local last_method = "none"
local auto_steps = 0

local function try_load_actions()
    loaded = false
    load_error = ""
    local ok, err = pcall(function() dofile(TABLE_FILE) end)
    if not ok then
        load_error = tostring(err)
        return false
    end
    if type(XIOI_V31_ACTIONS) ~= "table" then
        load_error = "XIOI_V31_ACTIONS missing"
        return false
    end
    actions = XIOI_V31_ACTIONS
    action_count = tonumber(XIOI_V31_ACTION_COUNT or 0) or 0
    TURNFRAMES = tonumber(XIOI_V31_TURNFRAMES or TURNFRAMES) or TURNFRAMES
    loaded = true
    return true
end

local function apply_joint(j, v)
    local ok = false
    if set_joint_state then
        ok = pcall(function() set_joint_state(0, j, v) end)
        if ok then
            last_method = "set_joint_state(0,j,v)"
        else
            ok = pcall(function() set_joint_state(j, v) end)
            if ok then last_method = "set_joint_state(j,v)" end
        end
    end
    if ok then motor_ok = motor_ok + 1 else motor_fail = motor_fail + 1 end
end

local function apply_action_for_frame(fr)
    if not loaded then return end
    local pairs = actions[fr]
    if pairs == nil then
        last_pairs = 0
        return
    end
    last_frame = fr
    last_pairs = #pairs
    for _, p in ipairs(pairs) do
        local j = tonumber(p[1])
        local v = tonumber(p[2])
        if j ~= nil and v ~= nil then apply_joint(j, v) end
    end
end

local function step_auto()
    -- Avance automatiquement comme nos anciens runners.
    if run_frames then
        pcall(function() run_frames(1) end)
        auto_steps = auto_steps + 1
    elseif step_game then
        pcall(function() step_game() end)
        auto_steps = auto_steps + 1
    end
end

local function on_enter_frame()
    tick = tick + 1
    if not loaded then try_load_actions() end
    local fr = math.floor((tick - 1) / TURNFRAMES) * TURNFRAMES
    apply_action_for_frame(fr)
    step_auto()
end

local function on_new_game()
    tick = 0
    motor_ok = 0
    motor_fail = 0
    last_pairs = 0
    last_frame = -1
    auto_steps = 0
    try_load_actions()
end

local function on_draw2d()
    set_color(1, 1, 1, 1)
    draw_text("Xioi Champion V31", 40, 40, 1)
    draw_text("loaded=" .. tostring(loaded) .. " actions=" .. tostring(action_count), 40, 60, 1)
    draw_text("tick=" .. tostring(tick) .. " last_frame=" .. tostring(last_frame) .. " pairs=" .. tostring(last_pairs), 40, 80, 1)
    draw_text("ok=" .. tostring(motor_ok) .. " fail=" .. tostring(motor_fail) .. " method=" .. tostring(last_method), 40, 100, 1)
    draw_text("auto=" .. tostring(auto_steps) .. " file=" .. TABLE_FILE, 40, 120, 1)
    if load_error ~= "" then draw_text("ERR=" .. load_error, 40, 145, 1) end
end

add_hook("new_game", "xioi_champion_v31_new_game", on_new_game)
add_hook("enter_frame", "xioi_champion_v31_enter_frame", on_enter_frame)
add_hook("draw2d", "xioi_champion_v31_draw", on_draw2d)
try_load_actions()
