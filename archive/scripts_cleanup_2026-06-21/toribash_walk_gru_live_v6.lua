echo("################################################")
echo("[Walk GRU Live V6] LUA LOADED - JSON WARMUP + GRU LOOP")
echo("################################################")

local CONFIG = {
    max_frames = 1200,
    frames_per_action = 5,
    warmup_frames = 0,
    fall_z = 4.25,
    head_ground_z = 4.4,
    hip_ground_z = 4.6,
    result_path = "toribashai_episode_result.json"
}

local AGENT_FILES = {
    "toribashai_agent_current.lua",
    "../data/script/toribashai_agent_current.lua"
}

local AGENT_JOINTS = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}

frame = 0
boot_ticks = 0
running = false
finished = false
started_physics = false
start_y = nil
best_y = nil
last_y = nil
stuck_frames = 0
agent_loaded = false
last_action_idx = 1

local function q(s)
    s = tostring(s or "")
    s = string.gsub(s, "\\", "\\\\")
    s = string.gsub(s, '"', '\\"')
    return '"' .. s .. '"'
end

local function write_result(score, reason)
    local f = io.open(CONFIG.result_path, "w")
    if not f then return end
    f:write("{\n")
    f:write('  "score": ' .. tostring(score) .. ",\n")
    f:write('  "reason": ' .. q(reason) .. ",\n")
    f:write('  "frames": ' .. tostring(frame) .. ",\n")
    f:write('  "start_y": ' .. tostring(start_y or 0) .. ",\n")
    f:write('  "last_y": ' .. tostring(last_y or 0) .. ",\n")
    f:write('  "best_y": ' .. tostring(best_y or 0) .. ",\n")
    f:write('  "agent": ' .. q(TORIBASHAI_AGENT and TORIBASHAI_AGENT.name or "missing") .. ",\n")
    f:write('  "generation": ' .. tostring(TORIBASHAI_AGENT and TORIBASHAI_AGENT.generation or -1) .. ",\n")
    f:write('  "candidate": ' .. tostring(TORIBASHAI_AGENT and TORIBASHAI_AGENT.agent_index or -1) .. "\n")
    f:write("}\n")
    f:close()
end

local function body_pos(player, body)
    local info = get_body_info(player, body)
    if not info then return nil end
    if info.pos then return info.pos.x or info.pos[1], info.pos.y or info.pos[2], info.pos.z or info.pos[3] end
    if info.x and info.y and info.z then return info.x, info.y, info.z end
    if info[1] and info[2] and info[3] then return info[1], info[2], info[3] end
    return nil
end

local function avg_body(ids)
    local sx, sy, sz, c = 0,0,0,0
    for _, id in ipairs(ids) do
        local ok, x, y, z = pcall(body_pos, 0, id)
        if ok and x and y and z then sx=sx+x; sy=sy+y; sz=sz+z; c=c+1 end
    end
    if c == 0 then return nil end
    return sx/c, sy/c, sz/c
end

local function tori_center()
    return avg_body({0,1,2,3,4,5,14,15})
end

local function head_z()
    local _, _, z = body_pos(0, 0)
    return z or 99
end

local function hip_z()
    local _, _, z = avg_body({4,5,14,15})
    return z or 99
end

local function load_agent()
    TORIBASHAI_AGENT = nil
    for _, path in ipairs(AGENT_FILES) do
        local ok, result = pcall(dofile, path)
        if ok and type(result) == "table" and result.actions then
            TORIBASHAI_AGENT = result
            echo("[Walk GRU V6] loaded agent via return: " .. path)
            break
        end
        if TORIBASHAI_AGENT and TORIBASHAI_AGENT.actions then
            echo("[Walk GRU V6] loaded global agent: " .. path)
            break
        end
    end
    if not TORIBASHAI_AGENT or not TORIBASHAI_AGENT.actions then
        echo("[Walk GRU V6] agent table missing")
        return false
    end
    CONFIG.frames_per_action = tonumber(TORIBASHAI_AGENT.frames_per_action or CONFIG.frames_per_action) or CONFIG.frames_per_action
    CONFIG.max_frames = tonumber(TORIBASHAI_AGENT.max_frames or CONFIG.max_frames) or CONFIG.max_frames
    echo("[Walk GRU V6] gen=" .. tostring(TORIBASHAI_AGENT.generation or 0) .. " agent=" .. tostring(TORIBASHAI_AGENT.agent_index or 0) .. " actions=" .. tostring(#TORIBASHAI_AGENT.actions))
    return true
end

local function hold_all()
    for j=0,19 do pcall(set_joint_state, 0, j, 3, true) end
end

local function action_index_for_frame()
    local idx = math.floor(frame / CONFIG.frames_per_action) + 1
    local actions = TORIBASHAI_AGENT.actions or {}
    if idx <= #actions then return idx end
    local loop_start = tonumber(TORIBASHAI_AGENT.loop_start_action or 1) or 1
    if loop_start < 1 then loop_start = 1 end
    if loop_start > #actions then loop_start = 1 end
    local loop_len = #actions - loop_start + 1
    if loop_len < 1 then return #actions end
    return loop_start + ((idx - loop_start) % loop_len)
end

local function apply_action()
    if not TORIBASHAI_AGENT or not TORIBASHAI_AGENT.actions then return end
    local idx = action_index_for_frame()
    last_action_idx = idx
    local action = TORIBASHAI_AGENT.actions[idx]
    if not action then return end
    local joints = TORIBASHAI_AGENT.control_joints or AGENT_JOINTS
    for i, j in ipairs(joints) do
        local v = tonumber(action[i] or action[tostring(i)] or 3) or 3
        if v > 0 then pcall(set_joint_state, 0, tonumber(j), v, true) end
    end
end

local function finish(reason)
    if finished then return end
    finished = true
    running = false
    local cy = last_y or start_y or 0
    local dy = 0
    if start_y and cy then dy = cy - start_y end
    local progress = math.abs(dy)
    local score = progress * 100.0 + frame * 1.0
    if reason ~= "max_frames" then score = score - 500.0 end
    echo("[Walk GRU V6] SCORE=" .. tostring(score) .. " reason=" .. tostring(reason) .. " frame=" .. tostring(frame) .. " action=" .. tostring(last_action_idx))
    write_result(score, reason)
    pcall(freeze_game)
end

local function start_physics_once()
    if started_physics then return end
    started_physics = true
    echo("[Walk GRU V6] auto-space / run")
    pcall(unfreeze_game)
    pcall(toggle_game_pause, false)
    pcall(step_game, false, false)
    pcall(run_frames, 1)
    pcall(run_frames, 10)
end

local function on_new_game()
    agent_loaded = load_agent()
    frame = 0; boot_ticks = 0; finished = false; started_physics = false; running = agent_loaded
    start_y = nil; best_y = nil; last_y = nil; stuck_frames = 0; last_action_idx = 1
    if not agent_loaded then pcall(freeze_game); return end
    hold_all()
end

local function on_enter_frame()
    if not running then return end
    frame = frame + 1
    local _, cy, cz = tori_center()
    if cy and not start_y then start_y = cy; best_y = cy end
    if cy then
        last_y = cy
        if not best_y or math.abs(cy - start_y) > math.abs(best_y - start_y) then best_y = cy; stuck_frames = 0 else stuck_frames = stuck_frames + 1 end
    end
    apply_action()
    apply_action()
    pcall(run_frames, 1)
    if frame >= CONFIG.max_frames then finish("max_frames"); return end
    if cz and cz < CONFIG.fall_z and frame > 35 then finish("fell_center"); return end
    if frame > 45 and head_z() < CONFIG.head_ground_z then finish("head_ground"); return end
    if frame > 60 and hip_z() < CONFIG.hip_ground_z then finish("hips_ground"); return end
end

local function on_draw2d()
    if running and not started_physics then
        boot_ticks = boot_ticks + 1
        if boot_ticks >= 1 then start_physics_once() end
    end
    draw_text("WalkGRU V6 gen=" .. tostring(TORIBASHAI_AGENT and TORIBASHAI_AGENT.generation or -1) .. " agent=" .. tostring(TORIBASHAI_AGENT and TORIBASHAI_AGENT.agent_index or -1), 40, 90, 1)
    draw_text("frame=" .. tostring(frame) .. " action=" .. tostring(last_action_idx) .. " auto-space=on", 40, 115, 1)
end

add_hook("new_game", "toribash_walk_gru_live_v6", on_new_game)
add_hook("enter_frame", "toribash_walk_gru_live_v6", on_enter_frame)
add_hook("draw2d", "toribash_walk_gru_live_v6", on_draw2d)
