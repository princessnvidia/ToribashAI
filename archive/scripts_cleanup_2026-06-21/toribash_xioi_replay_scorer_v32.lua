-- toribash_xioi_replay_scorer_v32.lua
-- Scorer passif pour replays complets xioi_master_final_v32.
-- Important : ce Lua NE CONTROLE PAS le Tori. Il observe le replay complet.

local RESULT_FILE = "xioi_replay_hybrid_v32_score.json"
local META_FILE = "xioi_replay_hybrid_v32_current.json"

local started = false
local start_y = nil
local best_y = nil
local last_y = nil
local frames_seen = 0
local upright_frames = 0
local low_frames = 0
local max_head_z = -9999
local min_head_z = 9999
local score_written = false
local current_gen = "?"
local current_candidate = "?"
local current_pop = "?"
local status = "loaded"

local function safe_read(path)
    local f = io.open(path, "r")
    if not f then return nil end
    local s = f:read("*all")
    f:close()
    return s
end

local function safe_write(path, text)
    local f = io.open(path, "w")
    if not f then return false end
    f:write(text)
    f:close()
    return true
end

local function load_meta()
    local txt = safe_read(META_FILE)
    if not txt then return end
    current_gen = txt:match('"generation"%s*:%s*(%d+)') or current_gen
    current_candidate = txt:match('"candidate"%s*:%s*(%d+)') or current_candidate
    current_pop = txt:match('"population"%s*:%s*(%d+)') or current_pop
end

local function body_pos(player, part)
    -- Plusieurs builds Toribash exposent des API différentes ; on tente plusieurs formes.
    if get_body_info then
        local ok, info = pcall(function() return get_body_info(player, part) end)
        if ok and type(info) == "table" then
            if info.pos then return info.pos[1], info.pos[2], info.pos[3] end
            if info.x and info.y and info.z then return info.x, info.y, info.z end
        end
    end
    if get_body_position then
        local ok, x, y, z = pcall(function() return get_body_position(player, part) end)
        if ok and x then return x, y, z end
    end
    if get_body_pos then
        local ok, x, y, z = pcall(function() return get_body_pos(player, part) end)
        if ok and x then return x, y, z end
    end
    return nil, nil, nil
end

local function estimate_core()
    -- Indices probables : head=0, chest/pecs/lumbar proches 1..4, hips/glutes 13..15 selon builds.
    local hx, hy, hz = body_pos(0, 0)
    local cx, cy, cz = body_pos(0, 1)
    local lx, ly, lz = body_pos(0, 2)
    local rx, ry, rz = body_pos(0, 3)
    local px, py, pz = nil, nil, nil

    local ys = {}
    local zs = {}
    local function add(x, y, z)
        if y and z then table.insert(ys, y); table.insert(zs, z) end
    end
    add(hx, hy, hz)
    add(cx, cy, cz)
    add(lx, ly, lz)
    add(rx, ry, rz)

    if #ys == 0 then return nil, nil, nil end
    local sy, sz = 0, 0
    for i=1,#ys do sy = sy + ys[i]; sz = sz + zs[i] end
    local mid_y = sy / #ys
    local mid_z = sz / #zs
    return mid_y, mid_z, hz
end

local function write_score(reason)
    if score_written then return end
    score_written = true
    local distance = 0
    if start_y and best_y then distance = best_y - start_y end
    local upright_bonus = upright_frames * 0.35
    local low_penalty = low_frames * 2.0
    local score = distance * 45.0 + upright_bonus - low_penalty + frames_seen * 0.05
    local txt = string.format('{"version":32,"generation":%s,"candidate":%s,"population":%s,"score":%.6f,"distance_y":%.6f,"frames":%d,"upright_frames":%d,"low_frames":%d,"max_head_z":%.6f,"min_head_z":%.6f,"reason":"%s"}', tostring(current_gen), tostring(current_candidate), tostring(current_pop), score, distance, frames_seen, upright_frames, low_frames, max_head_z, min_head_z, reason or "done")
    safe_write(RESULT_FILE, txt)
    status = "scored " .. tostring(score)
end

local function on_frame()
    load_meta()
    frames_seen = frames_seen + 1

    local y, z, head_z = estimate_core()
    if y then
        if not start_y then
            start_y = y
            best_y = y
        end
        last_y = y
        if y > best_y then best_y = y end
        if head_z then
            if head_z > max_head_z then max_head_z = head_z end
            if head_z < min_head_z then min_head_z = head_z end
            if head_z > 8.0 and z and z > 4.5 then upright_frames = upright_frames + 1 end
            if head_z < 4.0 then low_frames = low_frames + 1 end
        end
    else
        status = "no body api"
    end

    if frames_seen >= 420 then
        write_score("max_frames")
    end
end

local function on_draw()
    if draw_text then
        draw_text("xioi replay hybrid scorer v32", 40, 60, 1)
        draw_text("gen=" .. tostring(current_gen) .. " cand=" .. tostring(current_candidate) .. "/" .. tostring(current_pop), 40, 85, 1)
        draw_text("frames=" .. tostring(frames_seen) .. " status=" .. tostring(status), 40, 110, 1)
        local dist = 0
        if start_y and best_y then dist = best_y - start_y end
        draw_text(string.format("distY=%.3f upright=%d low=%d", dist, upright_frames, low_frames), 40, 135, 1)
    end
end

add_hook("enter_frame", "xioi_replay_hybrid_v32_frame", on_frame)
add_hook("draw2d", "xioi_replay_hybrid_v32_draw", on_draw)

load_meta()
status = "ready"
