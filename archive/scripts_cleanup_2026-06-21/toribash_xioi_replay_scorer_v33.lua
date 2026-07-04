-- toribash_xioi_replay_scorer_v33.lua
-- Scores full RPL candidates while Toribash plays them.
-- It does NOT control joints. It only observes.

local RESULT_FILE = "xioi_master_final_v33_score_result.txt"
local tick = 0
local start_y = nil
local best_y = 0
local last_y = 0
local min_head_z = 9999
local min_torso_z = 9999
local max_frames_alive = 0
local wrote = false

local function write_text(path, txt)
    local f = io.open(path, "w")
    if f then
        f:write(txt)
        f:close()
    end
end

local function get_body_pos_safe(player, part)
    if get_body_info then
        local ok, info = pcall(function() return get_body_info(player, part) end)
        if ok and type(info) == "table" then
            if info.pos then return info.pos.x or info.pos[1], info.pos.y or info.pos[2], info.pos.z or info.pos[3] end
            if info.x then return info.x, info.y, info.z end
        end
    end
    if get_body_pos then
        local ok, x, y, z = pcall(function() return get_body_pos(player, part) end)
        if ok and x then return x, y, z end
    end
    return nil, nil, nil
end

local function avg2(a, b)
    local ax, ay, az = get_body_pos_safe(0, a)
    local bx, by, bz = get_body_pos_safe(0, b)
    if ax and bx then return (ax + bx) / 2, (ay + by) / 2, (az + bz) / 2 end
    return nil, nil, nil
end

local function observe()
    tick = tick + 1

    -- Body indices may vary; these are best-effort and score still works approximately.
    local hx, hy, hz = get_body_pos_safe(0, 0)
    local sx, sy, sz = avg2(5, 8)
    local px, py, pz = avg2(13, 14)

    local y = nil
    local z = nil
    if py then y = py; z = pz elseif sy then y = sy; z = sz elseif hy then y = hy; z = hz end

    if y then
        if not start_y then start_y = y end
        last_y = y - start_y
        if last_y > best_y then best_y = last_y end
    end
    if hz and hz < min_head_z then min_head_z = hz end
    if z and z < min_torso_z then min_torso_z = z end

    if (not hz or hz > 2.0) and (not z or z > 1.0) then
        max_frames_alive = tick
    end

    if tick > 520 and not wrote then
        wrote = true
        local score = best_y * 100.0 + max_frames_alive * 0.25 + math.max(0, min_torso_z) * 2.0
        local txt = "score=" .. tostring(score) .. "\n" ..
                    "best_y=" .. tostring(best_y) .. "\n" ..
                    "last_y=" .. tostring(last_y) .. "\n" ..
                    "alive_frames=" .. tostring(max_frames_alive) .. "\n" ..
                    "min_head_z=" .. tostring(min_head_z) .. "\n" ..
                    "min_torso_z=" .. tostring(min_torso_z) .. "\n"
        write_text(RESULT_FILE, txt)
    end
end

local function draw()
    set_color(1, 1, 1, 1)
    draw_text("Xioi Replay Scorer V33", 40, 60, 1)
    draw_text("tick=" .. tostring(tick) .. " best_y=" .. string.format("%.3f", best_y), 40, 85, 1)
    draw_text("alive=" .. tostring(max_frames_alive) .. " minTorsoZ=" .. string.format("%.2f", min_torso_z), 40, 110, 1)
end

add_hook("enter_frame", "xioi_v33_observe", observe)
add_hook("draw2d", "xioi_v33_draw", draw)
