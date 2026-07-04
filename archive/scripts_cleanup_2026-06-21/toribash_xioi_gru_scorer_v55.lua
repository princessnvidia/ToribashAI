-- toribash_xioi_gru_scorer_v55.lua
-- ToribashAI V55 replay scorer: does NOT control joints.
-- It only auto-starts the loaded replay/match, displays gen/agent, and writes a score JSON.

local VERSION = "55"
local RESULT_FILE = "toribash_xioi_gru_score_v55.json"
local META_FILE = "toribash_xioi_gru_current_v55.txt"

local gen = "?"
local agent = "?"
local candidate = "?"
local started = false
local start_frame = nil
local last_frame = 0
local frames_alive = 0
local auto_ticks = 0
local wrote = false

local start_y = nil
local best_y = nil
local last_y = nil
local min_chest_z = 999999
local min_hip_z = 999999
local fall_frame = nil

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

local function load_meta()
    local s = read_file(META_FILE)
    if not s then return end
    gen = s:match("gen=([^\n]+)") or gen
    agent = s:match("agent=([^\n]+)") or agent
    candidate = s:match("candidate=([^\n]+)") or candidate
end

local function json_escape(s)
    s = tostring(s or "")
    s = s:gsub('\\', '\\\\')
    s = s:gsub('"', '\\"')
    return s
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
    return last_frame or 0
end

-- Body indexes in parsed RPL were usually:
-- 0 head, 1 chest-ish, 2 lumbar, 3 abs, 13/14 hips, 17/18/19/20 legs/feet depending source.
-- Toribash Lua body ids may differ; this scorer uses best-effort calls.
local function get_body_pos_best_effort(player, body)
    if get_body_info then
        local ok, info = pcall(function() return get_body_info(player, body) end)
        if ok and type(info) == "table" then
            local p = info.pos or info.position
            if type(p) == "table" then
                return tonumber(p.x or p[1]), tonumber(p.y or p[2]), tonumber(p.z or p[3])
            end
        end
    end
    if get_body_position then
        local ok, x, y, z = pcall(function() return get_body_position(player, body) end)
        if ok and x and y and z then return tonumber(x), tonumber(y), tonumber(z) end
    end
    if get_body_pos then
        local ok, x, y, z = pcall(function() return get_body_pos(player, body) end)
        if ok and x and y and z then return tonumber(x), tonumber(y), tonumber(z) end
    end
    return nil, nil, nil
end

local function sample_body()
    -- Try chest/lumbar/abs/hips and use the median-ish Y if possible.
    local candidates = {1, 2, 3, 13, 14}
    local ys = {}
    local zs = {}
    for _, b in ipairs(candidates) do
        local x, y, z = get_body_pos_best_effort(0, b)
        if y and z then
            table.insert(ys, y)
            table.insert(zs, z)
        end
    end
    if #ys == 0 then return nil, nil end
    table.sort(ys)
    table.sort(zs)
    local y = ys[math.floor((#ys + 1) / 2)]
    local z = zs[math.floor((#zs + 1) / 2)]
    return y, z
end

local function try_autostart()
    -- Auto-space / auto-play equivalent. Different Toribash builds expose different helpers.
    -- We call several safely; if none exist, user can still press Space once.
    if auto_ticks > 20 then return end
    auto_ticks = auto_ticks + 1
    if run_cmd then pcall(function() run_cmd(" ") end) end
    if run_frames then pcall(function() run_frames(1) end) end
    if step_game then pcall(function() step_game() end) end
    if play then pcall(function() play() end) end
end

local function compute_score(reason)
    local dist = 0
    if start_y and best_y then
        -- Direction may be negative on Xioi/assassin replay, so reward absolute forward progress from start to furthest body position.
        dist = math.abs(best_y - start_y)
    elseif start_y and last_y then
        dist = math.abs(last_y - start_y)
    end

    local upright_bonus = frames_alive * 0.8
    local dist_bonus = dist * 35.0
    local fall_penalty = fall_frame and 500 or 0
    local low_penalty = 0
    if min_chest_z < 7.0 then low_penalty = low_penalty + (7.0 - min_chest_z) * 40 end
    if min_hip_z < 5.5 then low_penalty = low_penalty + (5.5 - min_hip_z) * 45 end
    local score = dist_bonus + upright_bonus - fall_penalty - low_penalty

    return score, dist
end

local function write_result(reason)
    if wrote then return end
    wrote = true
    local score, dist = compute_score(reason or "done")
    local txt = string.format([[{
  "version": "%s",
  "gen": "%s",
  "agent": "%s",
  "candidate": "%s",
  "score": %.6f,
  "distance_y_body": %.6f,
  "frames_alive": %d,
  "last_frame": %d,
  "fall_frame": %s,
  "min_chest_z": %.6f,
  "min_hip_z": %.6f,
  "reason": "%s"
}
]], VERSION, json_escape(gen), json_escape(agent), json_escape(candidate), score, dist or 0, frames_alive, last_frame or 0, fall_frame and tostring(fall_frame) or "null", min_chest_z, min_hip_z, json_escape(reason or "done"))
    write_file(RESULT_FILE, txt)
end

local function update_score()
    load_meta()
    local fr = get_frame_safe()
    last_frame = fr

    if not started then
        try_autostart()
        if fr > 0 then
            started = true
            start_frame = fr
        end
    end

    local y, z = sample_body()
    if y then
        if not start_y then start_y = y end
        last_y = y
        if not best_y then best_y = y end
        if math.abs(y - start_y) > math.abs(best_y - start_y) then best_y = y end
    end
    if z then
        if z < min_chest_z then min_chest_z = z end
        if z < min_hip_z then min_hip_z = z end
    end

    if started then frames_alive = frames_alive + 1 end

    -- Conservative fall heuristic: if torso/hips are very low after the opening.
    if started and fr > 120 and z and z < 4.5 and not fall_frame then
        fall_frame = fr
        write_result("fall_low_body")
    end

    if fr >= 1100 then
        write_result("max_frames")
    end
end

local function draw_overlay()
    if set_color then set_color(1, 1, 1, 1) end
    if draw_text then
        draw_text("Xioi GRU V55 scorer", 40, 60, 1)
        draw_text("gen=" .. tostring(gen) .. " agent=" .. tostring(agent), 40, 85, 1)
        draw_text("candidate=" .. tostring(candidate), 40, 110, 1)
        draw_text("frame=" .. tostring(last_frame) .. " alive=" .. tostring(frames_alive), 40, 135, 1)
        draw_text("auto-space ticks=" .. tostring(auto_ticks), 40, 160, 1)
        if start_y and last_y then
            draw_text(string.format("bodyY %.2f -> %.2f", start_y, last_y), 40, 185, 1)
        end
    end
end

if add_hook then
    add_hook("enter_frame", "toribashai_xioi_gru_v55_update", update_score)
    add_hook("draw2d", "toribashai_xioi_gru_v55_draw", draw_overlay)
end

load_meta()
write_file(RESULT_FILE, "{\"status\":\"loaded\",\"version\":\"55\"}\n")
