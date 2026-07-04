-- ToribashAI walk_gru_live_runner_v3.lua
-- Live evolution runner inspired by upright_runner_v18.
-- Loads a frame-indexed agent table and applies exact JOINT pairs.

echo("################################################")
echo("[ToribashAI Walk GRU Live V3] LUA LOADED")
echo("################################################")

local CONFIG = {
    max_frames = 700,
    warmup_frames = 0,
    result_path = "toribash_walk_gru_live_result_v3.json",
    agent_file = "toribash_walk_gru_live_agent_v3.lua",
    auto_run_each_frame = true,
    target_forward_axis = "y",
    fall_head_z = 4.8,
    fall_chest_z = 4.6,
    fall_hip_z = 4.3,
    min_progress_after = 90,
    stale_window = 110,
}

frame = 0
boot_ticks = 0
running = false
finished = false
started_physics = false
last_applied_frame = -999999
last_pairs_count = 0
last_agent_name = "none"
last_gen = 0
last_agent_index = 0
last_score = 0
last_reason = "init"
apply_ok = 0
apply_fail = 0

start_x = nil
start_y = nil
start_chest_z = nil
best_progress = -999999
last_progress_frame = 0
progress_sum = 0
sample_count = 0
head_high_frames = 0
chest_high_frames = 0
hip_high_frames = 0
upright_sum = 0
hand_low_frames = 0
fall_frame = nil

TORIBASHAI_WALK_AGENT = nil

local function read_file(path)
    local f = io.open(path, "r")
    if not f then return nil end
    local s = f:read("*all")
    f:close()
    return s
end

local function write_file(path, text)
    local f = io.open(path, "w")
    if not f then
        echo("[Walk V3] cannot write " .. tostring(path))
        return false
    end
    f:write(text)
    f:close()
    return true
end

local function json_escape(s)
    s = tostring(s or "")
    s = s:gsub('\\', '\\\\')
    s = s:gsub('"', '\\"')
    s = s:gsub('\n', ' ')
    return s
end

local function load_agent()
    TORIBASHAI_WALK_AGENT = nil
    TORIBASHAI_AGENT = nil

    local ok, returned = pcall(function()
        return dofile(CONFIG.agent_file)
    end)

    if not ok then
        echo("[Walk V3] AGENT LOAD ERROR: " .. tostring(returned))
        return false
    end

    local agent = returned or TORIBASHAI_WALK_AGENT or TORIBASHAI_AGENT
    if not agent then
        echo("[Walk V3] agent table missing")
        return false
    end

    TORIBASHAI_WALK_AGENT = agent
    if not TORIBASHAI_WALK_AGENT.actions then
        echo("[Walk V3] agent actions missing")
        return false
    end

    last_agent_name = tostring(TORIBASHAI_WALK_AGENT.name or "unnamed")
    last_gen = tonumber(TORIBASHAI_WALK_AGENT.generation or 0) or 0
    last_agent_index = tonumber(TORIBASHAI_WALK_AGENT.agent_index or 0) or 0
    echo("[Walk V3] AGENT LOADED: " .. last_agent_name)
    echo("[Walk V3] gen=" .. tostring(last_gen) .. " agent=" .. tostring(last_agent_index) .. " actions=" .. tostring(#TORIBASHAI_WALK_AGENT.actions))
    return true
end

local function clamp_joint_value(v)
    v = tonumber(v) or 3
    if v < 1 then v = 1 end
    if v > 4 then v = 4 end
    return v
end

local function set_joint_safe(j, v)
    j = tonumber(j)
    v = clamp_joint_value(v)
    if not j then return false end

    local ok = pcall(function()
        set_joint_state(0, j, v)
    end)
    if not ok then
        ok = pcall(function()
            set_joint_state(j, v)
        end)
    end
    return ok
end

local function hold_all()
    for j = 0, 19 do
        set_joint_safe(j, 3)
    end
end

local function body_pos(player, body)
    local info = get_body_info(player, body)
    if not info then return nil end
    if info.pos then
        return info.pos.x or info.pos[1], info.pos.y or info.pos[2], info.pos.z or info.pos[3]
    end
    if info.x and info.y and info.z then return info.x, info.y, info.z end
    if info[1] and info[2] and info[3] then return info[1], info[2], info[3] end
    return nil
end

local function avg_body(ids)
    local sx, sy, sz, c = 0, 0, 0, 0
    for _, id in ipairs(ids) do
        local ok, x, y, z = pcall(body_pos, 0, id)
        if ok and x and y and z then
            sx = sx + x; sy = sy + y; sz = sz + z; c = c + 1
        end
    end
    if c == 0 then return nil end
    return sx / c, sy / c, sz / c
end

local function get_head()
    return body_pos(0, 0)
end

local function get_chest()
    return avg_body({1, 2, 3})
end

local function get_hips()
    return avg_body({4, 5, 14, 15})
end

local function get_lowest_hand_z()
    local lowest = nil
    for _, id in ipairs({8,9,10,11,12,13}) do
        local x,y,z = body_pos(0, id)
        if z and (lowest == nil or z < lowest) then lowest = z end
    end
    return lowest or 99
end

local function progress_from(chx, chy)
    if not start_y then return 0 end
    -- Our Xioi/assassin replay moves toward negative Y; progress is start_y - current_y.
    return start_y - chy
end

local function build_action_index()
    local agent = TORIBASHAI_WALK_AGENT
    if not agent then return end
    agent.action_by_frame = {}
    for _, a in ipairs(agent.actions or {}) do
        local fr = tonumber(a.frame or 0) or 0
        agent.action_by_frame[fr] = a.pairs or {}
    end
end

local function apply_action_for_frame(fr)
    local agent = TORIBASHAI_WALK_AGENT
    if not agent or not agent.actions then return end
    local loop_length = tonumber(agent.loop_length or 0) or 0
    local action_frame = fr
    if loop_length > 0 and fr >= (agent.loop_start_frame or 0) then
        local loop_start = tonumber(agent.loop_start_frame or 0) or 0
        action_frame = loop_start + ((fr - loop_start) % loop_length)
    end

    local pairs = nil
    if agent.action_by_frame then
        pairs = agent.action_by_frame[action_frame]
    end
    if not pairs then
        return
    end

    last_applied_frame = action_frame
    last_pairs_count = #pairs
    for _, p in ipairs(pairs) do
        local j = p[1]
        local v = p[2]
        if set_joint_safe(j, v) then
            apply_ok = apply_ok + 1
        else
            apply_fail = apply_fail + 1
        end
    end
end

local function reset_stats()
    frame = 0
    boot_ticks = 0
    running = false
    finished = false
    started_physics = false
    last_applied_frame = -999999
    last_pairs_count = 0
    apply_ok = 0
    apply_fail = 0
    start_x = nil
    start_y = nil
    start_chest_z = nil
    best_progress = -999999
    last_progress_frame = 0
    progress_sum = 0
    sample_count = 0
    head_high_frames = 0
    chest_high_frames = 0
    hip_high_frames = 0
    upright_sum = 0
    hand_low_frames = 0
    fall_frame = nil
    last_score = 0
    last_reason = "running"
end

local function finish_run(reason)
    if finished then return end
    finished = true
    running = false
    last_reason = reason or "done"

    local hx, hy, hz = get_head()
    local cx, cy, cz = get_chest()
    local px, py, pz = get_hips()
    local progress = 0
    if cx and cy then progress = progress_from(cx, cy) end

    local upright_ratio = 0
    if sample_count > 0 then upright_ratio = upright_sum / sample_count end

    local score = 0
    score = score + progress * 120.0
    score = score + best_progress * 60.0
    score = score + upright_ratio * 120.0
    score = score + chest_high_frames * 0.35
    score = score + hip_high_frames * 0.30
    score = score - hand_low_frames * 2.5

    if reason == "fell" or reason == "stale" or reason == "bad_body" then
        score = score - 250.0
    end

    last_score = score

    local txt = "{\n"
    txt = txt .. '  "version": 3,\n'
    txt = txt .. '  "agent": "' .. json_escape(last_agent_name) .. '",\n'
    txt = txt .. '  "generation": ' .. tostring(last_gen) .. ',\n'
    txt = txt .. '  "agent_index": ' .. tostring(last_agent_index) .. ',\n'
    txt = txt .. '  "score": ' .. string.format("%.6f", score) .. ',\n'
    txt = txt .. '  "reason": "' .. json_escape(reason) .. '",\n'
    txt = txt .. '  "frames": ' .. tostring(frame) .. ',\n'
    txt = txt .. '  "progress": ' .. string.format("%.6f", progress) .. ',\n'
    txt = txt .. '  "best_progress": ' .. string.format("%.6f", best_progress) .. ',\n'
    txt = txt .. '  "upright_ratio": ' .. string.format("%.6f", upright_ratio) .. ',\n'
    txt = txt .. '  "apply_ok": ' .. tostring(apply_ok) .. ',\n'
    txt = txt .. '  "apply_fail": ' .. tostring(apply_fail) .. ',\n'
    txt = txt .. '  "last_action_frame": ' .. tostring(last_applied_frame) .. '\n'
    txt = txt .. "}\n"

    write_file(CONFIG.result_path, txt)
    echo("[Walk V3] finished reason=" .. tostring(reason) .. " score=" .. string.format("%.2f", score))
end

local function sample_state()
    local hx, hy, hz = get_head()
    local cx, cy, cz = get_chest()
    local px, py, pz = get_hips()
    if not cx or not cy or not cz or not px or not py or not pz then return end

    if start_y == nil then
        start_x = cx
        start_y = cy
        start_chest_z = cz
        best_progress = 0
        last_progress_frame = frame
    end

    local progress = progress_from(cx, cy)
    if progress > best_progress then
        best_progress = progress
        last_progress_frame = frame
    end

    progress_sum = progress_sum + progress
    sample_count = sample_count + 1

    local upright = 0
    if hz and cz and pz then
        local torso_height = hz - pz
        local chest_height = cz - pz
        upright = math.max(0, math.min(1.5, torso_height / 3.0)) + math.max(0, math.min(1.0, chest_height / 1.5))
        upright_sum = upright_sum + upright
        if hz > 6.7 then head_high_frames = head_high_frames + 1 end
        if cz > 5.8 then chest_high_frames = chest_high_frames + 1 end
        if pz > 4.8 then hip_high_frames = hip_high_frames + 1 end
    end

    local hand_z = get_lowest_hand_z()
    if hand_z < 4.9 then hand_low_frames = hand_low_frames + 1 end

    if hz and cz and pz then
        if hz < CONFIG.fall_head_z or cz < CONFIG.fall_chest_z or pz < CONFIG.fall_hip_z then
            fall_frame = frame
            finish_run("fell")
            return
        end
    end

    if frame > CONFIG.min_progress_after and frame - last_progress_frame > CONFIG.stale_window then
        finish_run("stale")
        return
    end
end

local function on_new_game()
    reset_stats()
    load_agent()
    build_action_index()
    hold_all()
end

local function on_draw2d()
    set_color(1, 0.82, 1, 1)
    draw_text("ToribashAI Walk GRU Live V3", 35, 55, 1)
    draw_text("gen=" .. tostring(last_gen) .. " agent=" .. tostring(last_agent_index) .. " name=" .. tostring(last_agent_name), 35, 78, 1)
    draw_text("frame=" .. tostring(frame) .. " action=" .. tostring(last_applied_frame) .. " pairs=" .. tostring(last_pairs_count), 35, 101, 1)
    draw_text("ok=" .. tostring(apply_ok) .. " fail=" .. tostring(apply_fail) .. " score=" .. string.format("%.1f", last_score), 35, 124, 1)
    draw_text("reason=" .. tostring(last_reason) .. " running=" .. tostring(running), 35, 147, 1)

    -- Auto-start like old reliable runners: after script load/reset, push physics forward.
    if not finished and not started_physics then
        boot_ticks = boot_ticks + 1
        if boot_ticks >= 2 then
            started_physics = true
            running = true
            pcall(function() run_frames(1) end)
            pcall(function() run_frames(10) end)
        end
    end
end

local function on_enter_frame()
    if finished then return end

    if not TORIBASHAI_WALK_AGENT then
        load_agent()
        build_action_index()
    end

    apply_action_for_frame(frame)
    sample_state()

    frame = frame + 1

    if frame >= CONFIG.max_frames then
        finish_run("max_frames")
        return
    end

    if CONFIG.auto_run_each_frame and not finished then
        pcall(function() run_frames(1) end)
    end
end

add_hook("new_game", "toribashai_walk_gru_live_runner_v3_new", on_new_game)
add_hook("draw2d", "toribashai_walk_gru_live_runner_v3_draw", on_draw2d)
add_hook("enter_frame", "toribashai_walk_gru_live_runner_v3_frame", on_enter_frame)

load_agent()
build_action_index()
hold_all()
