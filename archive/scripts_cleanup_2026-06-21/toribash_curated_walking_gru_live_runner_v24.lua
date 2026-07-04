-- toribash_curated_walking_gru_live_runner_v24.lua
-- Live runner pour curated walking GRU V24.

local INPUT_FILE = "curated_walking_gru_v24_live_actions_current.json"
local RESULT_FILE = "../data/script/curated_walking_gru_v24_live_result.json"

local actions = {}
local action_by_frame = {}
local loaded = false
local tick = 0
local motor_ok = 0
local motor_fail = 0
local last_pairs = 0
local last_frame = -1
local status = "boot"
local warmup_frames = 18
local max_tick = 900

local function read_file(path)
    local f = io.open(path, "r")
    if not f then return nil end
    local s = f:read("*all")
    f:close()
    return s
end

local function parse_actions_json(txt)
    local parsed = {}
    if not txt then return parsed end

    -- Parser robuste pour blocs { "frame": N, "pairs": [[j,v], ...] }
    for block in txt:gmatch('%b{}') do
        local frame = block:match('"frame"%s*:%s*(%d+)')
        local pairs_part = block:match('"pairs"%s*:%s*(%b[])')
        if frame and pairs_part then
            local pairs = {}
            for j, v in pairs_part:gmatch('%[%s*(%d+)%s*,%s*(%d+)%s*%]') do
                table.insert(pairs, { tonumber(j), tonumber(v) })
            end
            table.insert(parsed, { frame = tonumber(frame), pairs = pairs })
        end
    end
    return parsed
end

local function load_actions()
    local txt = read_file(INPUT_FILE)
    actions = parse_actions_json(txt)
    action_by_frame = {}
    for _, a in ipairs(actions) do
        action_by_frame[tonumber(a.frame) or 0] = a.pairs or {}
    end
    loaded = (#actions > 0)
    status = loaded and "loaded" or "no_actions"
end

local function write_result(done)
    local f = io.open(RESULT_FILE, "w")
    if not f then return end
    f:write(string.format('{"version":"24","done":%s,"tick":%d,"motor_ok":%d,"motor_fail":%d,"last_frame":%d,"last_pairs":%d}\n',
        done and "true" or "false", tick, motor_ok, motor_fail, last_frame, last_pairs))
    f:close()
end

local function apply_joint(j, v)
    local ok = false
    if set_joint_state then
        ok = pcall(function() set_joint_state(0, j, v) end)
        if not ok then ok = pcall(function() set_joint_state(j, v) end) end
    end
    if ok then motor_ok = motor_ok + 1 else motor_fail = motor_fail + 1 end
end

local function step()
    if not loaded then return end
    tick = tick + 1

    if tick < warmup_frames then
        return
    end

    local logical_frame = (tick - warmup_frames)
    local frame = logical_frame - (logical_frame % 5)
    local pairs = action_by_frame[frame]

    if pairs then
        last_frame = frame
        last_pairs = #pairs
        for _, p in ipairs(pairs) do
            local j = tonumber(p[1])
            local v = tonumber(p[2])
            if j and v then apply_joint(j, v) end
        end
    end

    if tick % 30 == 0 then
        write_result(false)
    end
    if tick >= max_tick then
        write_result(true)
    end
end

local function draw_overlay()
    set_color(1, 1, 1, 1)
    draw_text("Curated Walking GRU V24", 40, 80, 1)
    draw_text("status=" .. status .. " actions=" .. tostring(#actions), 40, 100, 1)
    draw_text("tick=" .. tostring(tick) .. " last_frame=" .. tostring(last_frame) .. " pairs=" .. tostring(last_pairs), 40, 120, 1)
    draw_text("ok=" .. tostring(motor_ok) .. " fail=" .. tostring(motor_fail), 40, 140, 1)
end

load_actions()
write_result(false)

add_hook("enter_frame", "toribashai_curated_walking_gru_v24_step", step)
add_hook("draw2d", "toribashai_curated_walking_gru_v24_draw", draw_overlay)
