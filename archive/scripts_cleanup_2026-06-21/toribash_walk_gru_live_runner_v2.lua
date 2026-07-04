-- toribash_walk_gru_live_runner_v2.lua
-- ToribashAI walk_gru_live_v2
-- Inspired by upright/recovery runners: global agent table, echo boot, auto run_frames, overlay gen/agent.

echo("################################################")
echo("[ToribashAI Walk GRU Live V2] LUA LOADED")
echo("################################################")

local CONFIG = {
    version = "walk_gru_live_v2",
    hook = "toribashai_walk_gru_live_v2",
    agent_path = "toribash_walk_gru_live_agent_v2.lua",
    result_path = "toribash_walk_gru_live_result_v2.json",
    meta_path = "toribash_walk_gru_live_meta_v2.txt",
    max_frames_default = 1100,
    fall_z = 4.25,
    warmup_frames = 20,
    auto_boot_ticks = 80,
}

-- Keep globals small, like our older stable runners.
frame = 0
boot_ticks = 0
running = false
finished = false
started_physics = false

agent = nil
actions = {}
next_action_idx = 1
applied_actions = 0
ok_count = 0
fail_count = 0

start_y = nil
last_y = nil
best_y = nil
min_body_z = 999999
fall_frame = nil
frames_alive = 0
last_action_text = "none"
last_load_status = "not_loaded"

meta_gen = "?"
meta_agent = "?"
meta_pop = "?"
meta_parent = "?"

local function read_file(path)
    local f = io.open(path, "r")
    if not f then return nil end
    local s = f:read("*all")
    f:close()
    return s
end

local function write_file(path, txt)
    local f = io.open(path, "w")
    if not f then return false end
    f:write(txt)
    f:close()
    return true
end

local function esc(s)
    s = tostring(s or "")
    s = s:gsub('\\', '\\\\')
    s = s:gsub('"', '\\"')
    return s
end

local function load_meta()
    local s = read_file(CONFIG.meta_path)
    if not s then return end
    meta_gen = s:match("gen=([^\n]+)") or meta_gen
    meta_agent = s:match("agent=([^\n]+)") or meta_agent
    meta_pop = s:match("population=([^\n]+)") or meta_pop
    meta_parent = s:match("parent_score=([^\n]+)") or meta_parent
end

local function normalize_actions(data)
    local list = {}
    if type(data) == "table" then
        if type(data.actions) == "table" then list = data.actions end
        if #list == 0 and type(data.commands) == "table" then list = data.commands end
    end
    table.sort(list, function(a, b)
        return tonumber(a.frame or 0) < tonumber(b.frame or 0)
    end)
    return list
end

local function load_agent()
    TORIBASHAI_WALK_GRU_AGENT = nil
    TORIBASHAI_AGENT = nil

    local ok, ret = pcall(dofile, CONFIG.agent_path)
    local data = nil

    if ok and type(ret) == "table" then
        data = ret
    elseif type(TORIBASHAI_WALK_GRU_AGENT) == "table" then
        data = TORIBASHAI_WALK_GRU_AGENT
    elseif type(TORIBASHAI_AGENT) == "table" then
        data = TORIBASHAI_AGENT
    end

    if type(data) ~= "table" then
        agent = { name = "agent_table_missing", gen = "?", agent_id = "?", population = "?", max_frame = CONFIG.max_frames_default, actions = {} }
        actions = {}
        last_load_status = "agent table missing"
        echo("[Walk GRU V2] AGENT TABLE MISSING")
        if not ok then echo(tostring(ret)) end
        return false
    end

    agent = data
    actions = normalize_actions(data)
    last_load_status = "loaded " .. tostring(#actions) .. " actions"
    echo("[Walk GRU V2] agent loaded: " .. tostring(agent.name or "?") .. " actions=" .. tostring(#actions))
    return #actions > 0
end

local function get_frame_safe()
    if get_world_state then
        local ok, ws = pcall(get_world_state)
        if ok and type(ws) == "table" then
            if ws.match_frame then return tonumber(ws.match_frame) or 0 end
            if ws.frame then return tonumber(ws.frame) or 0 end
        end
    end
    if get_game_frame then
        local ok, fr = pcall(get_game_frame)
        if ok and fr then return tonumber(fr) or 0 end
    end
    if get_frame then
        local ok, fr = pcall(get_frame)
        if ok and fr then return tonumber(fr) or 0 end
    end
    return frame or 0
end

local function get_body_pos_best(player, body)
    if get_body_info then
        local ok, info = pcall(get_body_info, player, body)
        if ok and type(info) == "table" then
            if type(info.pos) == "table" then
                return tonumber(info.pos.x or info.pos[1]), tonumber(info.pos.y or info.pos[2]), tonumber(info.pos.z or info.pos[3])
            end
            if info.x and info.y and info.z then return tonumber(info.x), tonumber(info.y), tonumber(info.z) end
        end
    end
    if get_body_pos then
        local ok, x, y, z = pcall(get_body_pos, player, body)
        if ok and x and y and z then return tonumber(x), tonumber(y), tonumber(z) end
    end
    return nil, nil, nil
end

local function sample_body_center()
    -- Best effort: chest/lumbar/abs/hips-ish body ids used through the project.
    local ids = {1, 2, 3, 13, 14}
    local sx, sy, sz, n = 0, 0, 0, 0
    for _, id in ipairs(ids) do
        local x, y, z = get_body_pos_best(0, id)
        if x and y and z then
            sx = sx + x; sy = sy + y; sz = sz + z; n = n + 1
        end
    end
    if n > 0 then return sx / n, sy / n, sz / n end
    return nil, nil, nil
end

local function set_joint_best(j, v)
    local ok = false
    if set_joint_state then
        ok = pcall(function() set_joint_state(0, j, v) end)
        if not ok then ok = pcall(function() set_joint_state(j, v) end) end
    end
    if not ok and set_joint then
        ok = pcall(function() set_joint(0, j, v) end)
        if not ok then ok = pcall(function() set_joint(j, v) end) end
    end
    if ok then ok_count = ok_count + 1 else fail_count = fail_count + 1 end
    return ok
end

local function apply_action(cmd)
    if type(cmd) ~= "table" then return end
    local pairs = cmd.pairs or {}
    last_action_text = "f" .. tostring(cmd.frame or "?") .. " pairs=" .. tostring(#pairs)
    for _, p in ipairs(pairs) do
        local j = tonumber(p[1])
        local v = tonumber(p[2])
        if j and v and v ~= 0 then set_joint_best(j, v) end
    end
    applied_actions = applied_actions + 1
end

local function auto_step()
    -- This is the old-runner behavior that avoids pressing space repeatedly.
    if boot_ticks < CONFIG.auto_boot_ticks then
        boot_ticks = boot_ticks + 1
        if run_frames then pcall(function() run_frames(1) end) end
        if step_game then pcall(function() step_game() end) end
        if run_cmd then
            pcall(function() run_cmd(" ") end)
            pcall(function() run_cmd("play") end)
        end
    else
        if run_frames then pcall(function() run_frames(1) end) end
    end
end

local function compute_score(reason)
    local dist = 0
    if start_y and best_y then
        dist = math.abs(best_y - start_y)
    elseif start_y and last_y then
        dist = math.abs(last_y - start_y)
    end
    local alive = frames_alive * 0.75
    local distance = dist * 50.0
    local action = applied_actions * 0.15
    local fall = fall_frame and 800 or 0
    local low = 0
    if min_body_z < 4.8 then low = (4.8 - min_body_z) * 120 end
    return distance + alive + action - fall - low, dist
end

local function write_result(reason)
    if finished then return end
    finished = true
    local score, dist = compute_score(reason or "done")
    local max_frame = tonumber(agent and agent.max_frame or CONFIG.max_frames_default) or CONFIG.max_frames_default
    local txt = string.format([[{
  "version": "%s",
  "gen": "%s",
  "agent": "%s",
  "population": "%s",
  "score": %.6f,
  "distance_y_body": %.6f,
  "frames_alive": %d,
  "last_frame": %d,
  "max_frame": %d,
  "applied_actions": %d,
  "total_actions": %d,
  "ok_count": %d,
  "fail_count": %d,
  "fall_frame": %s,
  "min_body_z": %.6f,
  "load_status": "%s",
  "reason": "%s"
}
]], CONFIG.version, esc(meta_gen), esc(meta_agent), esc(meta_pop), score, dist, frames_alive, frame or 0, max_frame, applied_actions, #actions, ok_count, fail_count, fall_frame and tostring(fall_frame) or "null", min_body_z, esc(last_load_status), esc(reason or "done"))
    write_file(CONFIG.result_path, txt)
    echo("[Walk GRU V2] result score=" .. tostring(score) .. " reason=" .. tostring(reason))
end

local function reset_local_state()
    frame = 0
    boot_ticks = 0
    running = false
    finished = false
    started_physics = false
    next_action_idx = 1
    applied_actions = 0
    ok_count = 0
    fail_count = 0
    start_y = nil
    last_y = nil
    best_y = nil
    min_body_z = 999999
    fall_frame = nil
    frames_alive = 0
    last_action_text = "none"
    load_meta()
    load_agent()
end

local function on_new_game()
    reset_local_state()
end

local function on_enter_frame()
    if not agent then load_agent() end
    load_meta()

    frame = get_frame_safe()
    auto_step()

    if frame > 0 then started_physics = true end

    while actions[next_action_idx] and tonumber(actions[next_action_idx].frame or 0) <= frame do
        apply_action(actions[next_action_idx])
        next_action_idx = next_action_idx + 1
    end

    local x, y, z = sample_body_center()
    if y then
        if not start_y then start_y = y end
        last_y = y
        if not best_y then best_y = y end
        if math.abs(y - start_y) > math.abs(best_y - start_y) then best_y = y end
    end
    if z and z < min_body_z then min_body_z = z end

    if started_physics then frames_alive = frames_alive + 1 end

    if started_physics and frame > 140 and z and z < CONFIG.fall_z and not fall_frame then
        fall_frame = frame
        write_result("fall_low_body")
    end

    local max_frame = tonumber(agent and agent.max_frame or CONFIG.max_frames_default) or CONFIG.max_frames_default
    if frame >= max_frame then write_result("max_frames") end
end

local function draw_overlay()
    if set_color then set_color(1, 1, 1, 1) end
    if draw_text then
        draw_text("ToribashAI Walk GRU Live V2", 40, 55, 1)
        draw_text("gen=" .. tostring(meta_gen) .. " agent=" .. tostring(meta_agent) .. "/" .. tostring(meta_pop), 40, 80, 1)
        draw_text("frame=" .. tostring(frame) .. " actions=" .. tostring(applied_actions) .. "/" .. tostring(#actions), 40, 105, 1)
        draw_text("ok=" .. tostring(ok_count) .. " fail=" .. tostring(fail_count) .. " boot=" .. tostring(boot_ticks), 40, 130, 1)
        draw_text("load=" .. tostring(last_load_status), 40, 155, 1)
        draw_text("last=" .. tostring(last_action_text), 40, 180, 1)
        if start_y and last_y then draw_text(string.format("bodyY %.2f -> %.2f", start_y, last_y), 40, 205, 1) end
    end
end

reset_local_state()

if add_hook then
    add_hook("new_game", CONFIG.hook .. "_new", on_new_game)
    add_hook("enter_frame", CONFIG.hook .. "_step", on_enter_frame)
    add_hook("draw2d", CONFIG.hook .. "_draw", draw_overlay)
end
