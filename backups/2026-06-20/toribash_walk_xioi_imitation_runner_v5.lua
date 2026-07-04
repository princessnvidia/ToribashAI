-- toribash_walk_xioi_imitation_runner_v5.lua
-- Branche walk_xioi_imitation V5
-- Score: imitation de trajectoire Xioi + équilibre + avance
-- Lit:
--   walk_xioi_imitation_champion_v1.json
--   xioi_reference_trajectory_v1.json
-- Écrit:
--   toribashai_episode_result.json

echo("################################################")
echo("[walk_xioi_imitation_v5] LUA LOADED - CACHE SAFE")
echo("################################################")

local AGENT_PATH = "walk_xioi_imitation_champion_v1.json"
local REFERENCE_PATH = "xioi_reference_trajectory_v1.json"
local RESULT_PATH = "toribashai_episode_result_walk_xioi_imitation_v5.json"

local PLAYER = 0
local LOOP_LENGTH = 428
local MAX_FRAMES = 500

-- Ground calibrated for ToribashAI flat mod, close to upright_v18 logic.
local HEAD_GROUND_Z = 6.05
local HIP_GROUND_Z = 5.55
local SHOULDER_GROUND_Z = 5.85
local HAND_GROUND_Z = 5.65
local TORSO_GROUND_Z = 5.75

-- Body ids used for imitation snapshots.
local BODY_IDS = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
local BODY_WEIGHTS = {
    [0] = 2.0,  -- head
    [1] = 2.0,  -- chest / torso-ish
    [2] = 2.0,
    [3] = 2.0,
    [4] = 1.6,
    [5] = 1.6,
    [6] = 1.2,
    [7] = 1.2,
    [8] = 1.2,
    [9] = 1.2,
    [10] = 0.8,
    [11] = 0.8,
    [12] = 1.4,
    [13] = 1.4,
}

local commands_by_frame = {}
local reference_by_frame = {}
local last_joint_states = {}

local frame = 0
local running = false
local finished = false

local agent_name = "unknown"
local display_gen = "?"
local display_candidate = "?"
local display_population = "?"

local start_x, start_y, start_z = nil, nil, nil
local best_x = -999999

local head_ground_frames = 0
local hip_ground_frames = 0
local shoulder_ground_frames = 0
local hand_ground_frames = 0
local torso_ground_frames = 0
local lateral_fall_frames = 0
local stalled_frames = 0

local imitation_score_sum = 0
local imitation_frames = 0
local imitation_error_sum = 0
local best_imitation_frame_score = -999999

local current_score_display = 0
local current_forward_display = 0
local current_error_display = 0

local function log(msg)
    echo("[walk_xioi_imitation_v5] " .. tostring(msg))
end

local function file_read_all(path)
    local f = io.open(path, "r")
    if not f then
        log("ERREUR lecture: " .. path)
        return nil
    end
    local content = f:read("*a")
    f:close()
    return content
end

local function extract_string(content, key, default)
    local v = content:match('"' .. key .. '"%s*:%s*"([^"]*)"')
    return v or default
end

local function extract_number(content, key, default)
    local v = content:match('"' .. key .. '"%s*:%s*([%-%d%.]+)')
    return v or default
end

local function body_pos(player, body)
    local info = get_body_info(player, body)
    if not info then return nil end

    if info.pos then
        return info.pos.x or info.pos[1], info.pos.y or info.pos[2], info.pos.z or info.pos[3]
    end

    if info.x and info.y and info.z then
        return info.x, info.y, info.z
    end

    if info[1] and info[2] and info[3] then
        return info[1], info[2], info[3]
    end

    return nil
end

local function avg_body(ids)
    local sx, sy, sz, c = 0, 0, 0, 0
    for _, id in ipairs(ids) do
        local ok, x, y, z = pcall(body_pos, PLAYER, id)
        if ok and x and y and z then
            sx = sx + x
            sy = sy + y
            sz = sz + z
            c = c + 1
        end
    end
    if c == 0 then return nil end
    return sx / c, sy / c, sz / c
end

local function get_center()
    return avg_body({0, 1, 2, 3, 4, 5})
end

local function get_head()
    return body_pos(PLAYER, 0)
end

local function get_hips()
    return avg_body({2, 3, 4, 5})
end

local function get_torso()
    return avg_body({1, 2, 3})
end

local function get_shoulders()
    return avg_body({6, 7, 8, 9})
end

local function get_lowest_hand_z(fallback)
    local lowest = fallback or 999999
    for _, id in ipairs({10, 11, 12, 13}) do
        local _, _, z = body_pos(PLAYER, id)
        if z and z < lowest then lowest = z end
    end
    return lowest
end

local function load_commands_from_json(path)
    local content = file_read_all(path)
    if not content then return false end

    agent_name = extract_string(content, "name", "unknown")
    display_gen = extract_number(content, "current_generation", "?")
    display_candidate = extract_number(content, "current_candidate", "?")
    display_population = extract_number(content, "population_size", "?")

    commands_by_frame = {}

    for frame_text, pairs_block in content:gmatch('"frame"%s*:%s*(%d+)%s*,%s*"pairs"%s*:%s*%[(.-)%]%s*}') do
        local command_frame = tonumber(frame_text)
        if command_frame then
            commands_by_frame[command_frame] = commands_by_frame[command_frame] or {}

            for joint_text, state_text in pairs_block:gmatch('%[%s*(%d+)%s*,%s*(%d+)%s*%]') do
                table.insert(commands_by_frame[command_frame], {
                    joint = tonumber(joint_text),
                    state = tonumber(state_text)
                })
            end
        end
    end

    local count_frames = 0
    local count_pairs = 0
    for _, pairs_for_frame in pairs(commands_by_frame) do
        count_frames = count_frames + 1
        count_pairs = count_pairs + #pairs_for_frame
    end

    log("agent JSON chargé: " .. count_frames .. " frames-commandes, " .. count_pairs .. " pairs")
    log("GEN " .. tostring(display_gen) .. " | CANDIDAT " .. tostring(display_candidate) .. "/" .. tostring(display_population))
    log("agent=" .. tostring(agent_name))

    return count_frames > 0
end

local function load_reference_from_json(path)
    local content = file_read_all(path)
    if not content then
        log("Référence absente: imitation désactivée")
        return false
    end

    reference_by_frame = {}

    -- Matches compact JSON produced by build_xioi_reference_trajectory.py.
    for frame_block in content:gmatch('{"frame"%s*:%s*%d+.-"bodies"%s*:%s*{.-}}') do
        local fnum = tonumber(frame_block:match('"frame"%s*:%s*(%d+)'))
        if fnum then
            reference_by_frame[fnum] = {}
            for bid, x, y, z in frame_block:gmatch('"(%d+)"%s*:%s*{%s*"x"%s*:%s*([%-%d%.]+)%s*,%s*"y"%s*:%s*([%-%d%.]+)%s*,%s*"z"%s*:%s*([%-%d%.]+)%s*}') do
                reference_by_frame[fnum][tonumber(bid)] = {
                    x = tonumber(x),
                    y = tonumber(y),
                    z = tonumber(z)
                }
            end
        end
    end

    local count = 0
    for _ in pairs(reference_by_frame) do count = count + 1 end
    log("référence chargée: " .. tostring(count) .. " frames")

    return count > 0
end

local function apply_pairs(command_frame)
    local pairs_for_frame = commands_by_frame[command_frame]
    if not pairs_for_frame then return end

    for _, pair in ipairs(pairs_for_frame) do
        if pair.joint and pair.state then
            if last_joint_states[pair.joint] ~= pair.state then
                set_joint_state(PLAYER, pair.joint, pair.state, false)
                last_joint_states[pair.joint] = pair.state
            end
        end
    end
end

local function imitation_frame_score(loop_frame)
    local ref = reference_by_frame[loop_frame]
    if not ref then return 0, 0 end

    local current_center_x, current_center_y, current_center_z = get_center()
    if not current_center_x then return 0, 0 end

    local ref_center = ref[-1]
    local ref_cx, ref_cy, ref_cz = 0, 0, 0
    if ref_center then
        ref_cx, ref_cy, ref_cz = ref_center.x or 0, ref_center.y or 0, ref_center.z or 0
    end

    local total_error = 0
    local total_weight = 0

    for _, body_id in ipairs(BODY_IDS) do
        local r = ref[body_id]
        if r then
            local x, y, z = body_pos(PLAYER, body_id)
            if x and y and z then
                -- Compare relative body layout, not absolute world travel.
                local dx = (x - current_center_x) - (r.x - ref_cx)
                local dy = (y - current_center_y) - (r.y - ref_cy)
                local dz = (z - current_center_z) - (r.z - ref_cz)
                local dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                local w = BODY_WEIGHTS[body_id] or 1.0
                total_error = total_error + dist * w
                total_weight = total_weight + w
            end
        end
    end

    if total_weight <= 0 then return 0, 0 end

    local avg_error = total_error / total_weight
    local score = math.max(0, 220.0 - avg_error * 38.0)

    return score, avg_error
end

local function write_result(score, reason, x, y, z, forward, lateral_drift, avg_imitation_error)
    local f = io.open(RESULT_PATH, "w")
    if not f then
        log("ERREUR écriture result")
        return
    end

    f:write(string.format(
        '{"score": %.4f, "reason": "%s", "frames": %d, "gen": "%s", "candidate": "%s", "population": "%s", "x": %.4f, "y": %.4f, "z": %.4f, "forward": %.4f, "lateral_drift": %.4f, "best_x": %.4f, "imitation_score_sum": %.4f, "imitation_frames": %d, "avg_imitation_error": %.4f, "best_imitation_frame_score": %.4f, "head_ground_frames": %d, "hip_ground_frames": %d, "shoulder_ground_frames": %d, "hand_ground_frames": %d, "torso_ground_frames": %d, "lateral_fall_frames": %d, "stalled_frames": %d}',
        score,
        tostring(reason),
        frame,
        tostring(display_gen),
        tostring(display_candidate),
        tostring(display_population),
        x or 0,
        y or 0,
        z or 0,
        forward or 0,
        lateral_drift or 0,
        best_x,
        imitation_score_sum,
        imitation_frames,
        avg_imitation_error or 0,
        best_imitation_frame_score,
        head_ground_frames,
        hip_ground_frames,
        shoulder_ground_frames,
        hand_ground_frames,
        torso_ground_frames,
        lateral_fall_frames,
        stalled_frames
    ))

    f:close()
end

local function finish_run(reason)
    if finished then return end

    finished = true
    running = false

    local x, y, z = get_center()
    x = x or start_x or 0
    y = y or start_y or 0
    z = z or start_z or 0

    local forward = x - (start_x or x)
    local lateral_drift = math.abs(y - (start_y or y))
    local avg_imitation_error = 0

    if imitation_frames > 0 then
        avg_imitation_error = imitation_error_sum / imitation_frames
    end

    local score = 0

    -- V3: imitation dominates, then stability, then forward.
    score = score + imitation_score_sum * 4.0
    score = score + frame * 4.0
    score = score + forward * 320.0
    score = score + best_x * 90.0

    score = score - avg_imitation_error * 500.0
    score = score - lateral_drift * 260.0

    score = score - hand_ground_frames * 220.0
    score = score - hip_ground_frames * 220.0
    score = score - shoulder_ground_frames * 260.0
    score = score - head_ground_frames * 340.0
    score = score - torso_ground_frames * 280.0
    score = score - lateral_fall_frames * 220.0
    score = score - stalled_frames * 60.0

    if reason == "max_frames" then score = score + 3500.0 else score = score - 4200.0 end
    if reason == "hand_ground" then score = score - 2400.0 end
    if reason == "hips_ground" then score = score - 2800.0 end
    if reason == "shoulder_ground" then score = score - 3200.0 end
    if reason == "head_ground" then score = score - 4200.0 end
    if reason == "torso_ground" then score = score - 3200.0 end
    if reason == "lateral_fall" then score = score - 2600.0 end
    if reason == "stalled_or_collapsed" then score = score - 1600.0 end

    current_score_display = score
    current_forward_display = forward
    current_error_display = avg_imitation_error

    log("================================================")
    log("GEN " .. tostring(display_gen) .. " | CANDIDAT " .. tostring(display_candidate) .. "/" .. tostring(display_population))
    log("SCORE = " .. tostring(score))
    log("reason=" .. tostring(reason) .. " frames=" .. tostring(frame) .. " forward=" .. tostring(forward) .. " lateral=" .. tostring(lateral_drift))
    log("imitation_error=" .. tostring(avg_imitation_error) .. " imitation_sum=" .. tostring(imitation_score_sum))
    log("================================================")

    write_result(score, reason, x, y, z, forward, lateral_drift, avg_imitation_error)

    pcall(function()
        freeze_game()
    end)
end

local function update_metrics(loop_frame)
    local x, y, z = get_center()
    if not x then return end

    if not start_x then
        start_x, start_y, start_z = x, y, z
    end

    local forward = x - start_x
    local lateral_drift = math.abs(y - start_y)

    if x > best_x then best_x = x end

    local imit_score, imit_error = imitation_frame_score(loop_frame)
    imitation_score_sum = imitation_score_sum + imit_score
    imitation_error_sum = imitation_error_sum + imit_error
    imitation_frames = imitation_frames + 1

    if imit_score > best_imitation_frame_score then
        best_imitation_frame_score = imit_score
    end

    current_forward_display = forward
    current_error_display = imit_error

    local _, _, head_z = get_head()
    local _, _, hip_z = get_hips()
    local _, _, shoulder_z = get_shoulders()
    local _, _, torso_z = get_torso()
    local hand_z = get_lowest_hand_z(z)

    if head_z and head_z < HEAD_GROUND_Z then head_ground_frames = head_ground_frames + 1 end
    if hip_z and hip_z < HIP_GROUND_Z then hip_ground_frames = hip_ground_frames + 1 end
    if shoulder_z and shoulder_z < SHOULDER_GROUND_Z then shoulder_ground_frames = shoulder_ground_frames + 1 end
    if torso_z and torso_z < TORSO_GROUND_Z then torso_ground_frames = torso_ground_frames + 1 end
    if hand_z and hand_z < HAND_GROUND_Z then hand_ground_frames = hand_ground_frames + 1 end

    if frame > 60 and lateral_drift > 2.5 then
        lateral_fall_frames = lateral_fall_frames + 1
    end

    if frame > 180 and forward < 0.35 then
        stalled_frames = stalled_frames + 1
    end

    if frame > 30 then
        if head_ground_frames > 2 then finish_run("head_ground") return end
        if shoulder_ground_frames > 3 then finish_run("shoulder_ground") return end
        if torso_ground_frames > 3 then finish_run("torso_ground") return end
        if hip_ground_frames > 4 then finish_run("hips_ground") return end
        if hand_ground_frames > 5 then finish_run("hand_ground") return end
    end

    if frame > 80 and lateral_fall_frames > 12 then
        finish_run("lateral_fall")
        return
    end

    if frame > 240 and stalled_frames > 60 then
        finish_run("stalled_or_collapsed")
        return
    end
end

local function start_physics()
    pcall(function() unfreeze_game() end)
    pcall(function() toggle_game_pause(false) end)
    pcall(function() run_frames(1) end)
end

local function on_new_game()
    frame = 0
    running = false
    finished = false
    last_joint_states = {}

    start_x, start_y, start_z = nil, nil, nil
    best_x = -999999

    head_ground_frames = 0
    hip_ground_frames = 0
    shoulder_ground_frames = 0
    hand_ground_frames = 0
    torso_ground_frames = 0
    lateral_fall_frames = 0
    stalled_frames = 0

    imitation_score_sum = 0
    imitation_frames = 0
    imitation_error_sum = 0
    best_imitation_frame_score = -999999

    current_score_display = 0
    current_forward_display = 0
    current_error_display = 0

    local ok = load_commands_from_json(AGENT_PATH)
    load_reference_from_json(REFERENCE_PATH)

    if not ok then
        log("runner désactivé")
        return
    end

    running = true
    log("runner actif")
    start_physics()
end

local function on_draw2d()
    pcall(function()
        set_color(1, 1, 1, 1)
        draw_text("walk_xioi_imitation V5 | GEN " .. tostring(display_gen) .. " | CAND " .. tostring(display_candidate) .. "/" .. tostring(display_population), 20, 60, 0)
        draw_text("frame " .. tostring(frame) .. " | forward " .. string.format("%.2f", current_forward_display) .. " | imit_err " .. string.format("%.2f", current_error_display), 20, 82, 0)
    end)
end

local function on_enter_frame()
    if not running or finished then return end

    local loop_frame = frame % LOOP_LENGTH

    apply_pairs(loop_frame)
    update_metrics(loop_frame)

    frame = frame + 1

    if frame >= MAX_FRAMES then
        finish_run("max_frames")
        return
    end

    if not finished then
        pcall(function() run_frames(1) end)
    end
end

remove_hooks("walk_xioi_imitation_runner_v1")
remove_hooks("walk_xioi_imitation_runner_v2")
remove_hooks("walk_xioi_imitation_runner_v5")
remove_hooks("walk_xioi_runner_v8")
remove_hooks("walk_xioi_runner_v7")
remove_hooks("walk_xioi_runner_v6")

add_hook("new_game", "walk_xioi_imitation_runner_v5", on_new_game)
add_hook("draw2d", "walk_xioi_imitation_runner_v5", on_draw2d)
add_hook("enter_frame", "walk_xioi_imitation_runner_v5", on_enter_frame)

on_new_game()
