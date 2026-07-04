-- toribash_xioi_master_final_v3.lua
-- Xioi Master Final V3
-- Lit xioi_master_final_v3_agent_current.lua (table Lua générée par Python),
-- applique les JOINT, ajoute un micro-pilote d'équilibre après frame 70,
-- puis écrit xioi_master_final_v3_result.json pour l'évolution.

local ACTION_FILE = "xioi_master_final_v3_agent_current.lua"
local RESULT_FILE = "xioi_master_final_v3_result.json"
local NAME = "xioi_master_final_v3"

local TURNFRAMES = 5
local CORRECT_FROM_FRAME = 70
local MAX_TICKS = 340
local AUTORUN = true

local agent = nil
local action_by_frame = {}
local loaded = false
local load_error = "not loaded"
local tick = 0
local motor_ok = 0
local motor_fail = 0
local last_pairs = 0
local last_method = "none"
local last_action_frame = -1
local result_written = false
local start_y = nil
local best_y = nil
local last_y = nil
local start_z = nil
local min_z = 9999
local max_z = -9999

local TORSO_IDS = {1, 2, 3, 4, 5, 8}      -- breast/chest/stomach/pecs-ish, fallback tolerant
local HIP_IDS = {3, 4, 13, 14}             -- lower torso / thighs-ish
local KNEE_LEG_IDS = {15, 16, 17, 18, 19}  -- legs/feet-ish

local function table_len(t)
    local n = 0
    if t then for _ in pairs(t) do n = n + 1 end end
    return n
end

local function safe_body_pos(part)
    -- Toribash Lua varie selon versions. On essaie plusieurs signatures.
    local ok, a, b, c

    if get_body_info then
        ok, a = pcall(function() return get_body_info(0, part) end)
        if ok and type(a) == "table" then
            if a.pos and type(a.pos) == "table" then
                return tonumber(a.pos.x or a.pos[1] or 0), tonumber(a.pos.y or a.pos[2] or 0), tonumber(a.pos.z or a.pos[3] or 0)
            end
            if a.x or a.y or a.z then
                return tonumber(a.x or 0), tonumber(a.y or 0), tonumber(a.z or 0)
            end
        end
    end

    if get_body_pos then
        ok, a, b, c = pcall(function() return get_body_pos(0, part) end)
        if ok and a then return tonumber(a) or 0, tonumber(b) or 0, tonumber(c) or 0 end
        ok, a, b, c = pcall(function() return get_body_pos(part) end)
        if ok and a then return tonumber(a) or 0, tonumber(b) or 0, tonumber(c) or 0 end
    end

    if get_body_position then
        ok, a, b, c = pcall(function() return get_body_position(0, part) end)
        if ok and a then return tonumber(a) or 0, tonumber(b) or 0, tonumber(c) or 0 end
        ok, a, b, c = pcall(function() return get_body_position(part) end)
        if ok and a then return tonumber(a) or 0, tonumber(b) or 0, tonumber(c) or 0 end
    end

    return nil, nil, nil
end

local function avg_pos(ids)
    local sx, sy, sz, n = 0, 0, 0, 0
    for _, id in ipairs(ids) do
        local x, y, z = safe_body_pos(id)
        if x and y and z then
            sx = sx + x; sy = sy + y; sz = sz + z; n = n + 1
        end
    end
    if n == 0 then return nil, nil, nil end
    return sx / n, sy / n, sz / n
end

local function safe_write(path, text)
    local ok, f = pcall(function() return io.open(path, "w") end)
    if not ok or not f then return false end

    local wrote = false
    if f.write then
        local ok1 = pcall(function() f:write(text) end)
        if ok1 then wrote = true end
        if not wrote then
            local ok2 = pcall(function() f.write(f, text) end)
            if ok2 then wrote = true end
        end
    end
    if f.close then pcall(function() f:close() end) end
    return wrote
end

local function json_escape(s)
    s = tostring(s or "")
    s = s:gsub('\\', '\\\\'):gsub('"', '\\"'):gsub('\n', '\\n')
    return s
end

local function set_joint(j, v)
    local ok = false
    if set_joint_state then
        ok = pcall(function() set_joint_state(0, j, v) end)
        if ok then
            motor_ok = motor_ok + 1
            last_method = "set_joint_state(0,j,v)"
            return true
        end
        ok = pcall(function() set_joint_state(j, v) end)
        if ok then
            motor_ok = motor_ok + 1
            last_method = "set_joint_state(j,v)"
            return true
        end
    end
    motor_fail = motor_fail + 1
    return false
end

local function load_agent()
    action_by_frame = {}
    loaded = false
    load_error = ""

    local ok, data = pcall(dofile, ACTION_FILE)
    if not ok then
        load_error = tostring(data)
        return
    end
    if type(data) ~= "table" then
        load_error = "action file did not return a table"
        return
    end

    agent = data
    TURNFRAMES = tonumber(agent.turnframes or TURNFRAMES) or TURNFRAMES
    MAX_TICKS = tonumber(agent.max_ticks or MAX_TICKS) or MAX_TICKS
    CORRECT_FROM_FRAME = tonumber(agent.correct_from_frame or CORRECT_FROM_FRAME) or CORRECT_FROM_FRAME

    for _, a in ipairs(agent.actions or {}) do
        local fr = tonumber(a.frame or 0) or 0
        action_by_frame[fr] = a.pairs or {}
    end

    loaded = true
    load_error = "ok"
end

local function apply_pairs(pairs)
    last_pairs = 0
    for _, p in ipairs(pairs or {}) do
        local j = tonumber(p[1])
        local v = tonumber(p[2])
        if j and v then
            last_pairs = last_pairs + 1
            set_joint(j, v)
        end
    end
end

local function balance_pilot(frame)
    -- V3: très doux, uniquement après frame 70.
    -- Les 3 premiers pas viennent du champion V30_23; on ne les remplace pas.
    if frame < CORRECT_FROM_FRAME then return end
    if not agent or agent.enable_pilot == false then return end

    local _, ty, tz = avg_pos(TORSO_IDS)
    local _, hy, hz = avg_pos(HIP_IDS)
    if not ty or not hy then return end

    local lean = ty - hy

    -- V3: correction beaucoup plus légère au début.
    -- 70-140: épaules seulement, seuil large, pas de jambes.
    if frame < 140 then
        if lean > 0.70 then
            set_joint(4, 2)
        elseif lean < -0.70 then
            set_joint(7, 2)
        end
        return
    end

    -- Après 140: contrepoids bras opposés plus actif.
    if lean > 0.45 then
        set_joint(4, 2); set_joint(7, 4)
    elseif lean < -0.45 then
        set_joint(4, 4); set_joint(7, 2)
    end

    -- Bassin/jambes seulement si le centre descend clairement.
    if hz and tz and hz < tz - 2.8 then
        set_joint(16, 3); set_joint(17, 3)
    end
end

local function update_score_track()
    local _, y, z = avg_pos(TORSO_IDS)
    if y and z then
        if not start_y then start_y = y; best_y = y; start_z = z end
        last_y = y
        if y > best_y then best_y = y end
        if z < min_z then min_z = z end
        if z > max_z then max_z = z end
    end
end

local function write_result(reason)
    if result_written then return end
    result_written = true

    local dy = 0
    if start_y and last_y then dy = last_y - start_y end
    local best_dy = 0
    if start_y and best_y then best_dy = best_y - start_y end

    local _, torso_y, torso_z = avg_pos(TORSO_IDS)
    local _, knees_y, knees_z = avg_pos(KNEE_LEG_IDS)
    torso_z = torso_z or 0
    knees_z = knees_z or 0

    -- Score demandé : distance parcourue sur Y par zone buste / au-dessus des genoux.
    local above_knees = torso_z - knees_z
    local score = (best_dy * 100.0) + (dy * 60.0) + (tick * 0.2) + (above_knees * 8.0)
    if dy < 0 then score = score + dy * 180.0 end
    if above_knees < 1.0 then score = score - 250.0 end
    if min_z < 0.5 then score = score - 150.0 end

    local run_id = agent and agent.run_id or "unknown"
    local generation = agent and agent.generation or -1
    local candidate = agent and agent.candidate or -1

    local text = "{\n" ..
        "  \"name\": \"xioi_master_final_v3\",\n" ..
        "  \"run_id\": \"" .. json_escape(run_id) .. "\",\n" ..
        "  \"generation\": " .. tostring(generation) .. ",\n" ..
        "  \"candidate\": " .. tostring(candidate) .. ",\n" ..
        "  \"score\": " .. tostring(score) .. ",\n" ..
        "  \"dy\": " .. tostring(dy) .. ",\n" ..
        "  \"best_dy\": " .. tostring(best_dy) .. ",\n" ..
        "  \"torso_z\": " .. tostring(torso_z) .. ",\n" ..
        "  \"knees_z\": " .. tostring(knees_z) .. ",\n" ..
        "  \"above_knees\": " .. tostring(above_knees) .. ",\n" ..
        "  \"ticks\": " .. tostring(tick) .. ",\n" ..
        "  \"motor_ok\": " .. tostring(motor_ok) .. ",\n" ..
        "  \"motor_fail\": " .. tostring(motor_fail) .. ",\n" ..
        "  \"reason\": \"" .. json_escape(reason) .. "\"\n" ..
        "}\n"
    safe_write(RESULT_FILE, text)
end

local function reset_state()
    tick = 0
    motor_ok = 0
    motor_fail = 0
    last_pairs = 0
    last_method = "none"
    last_action_frame = -1
    result_written = false
    start_y = nil
    best_y = nil
    last_y = nil
    start_z = nil
    min_z = 9999
    max_z = -9999
    load_agent()
end

reset_state()

add_hook("new_game", NAME .. "_newgame", function()
    reset_state()
    -- V3: relance plus vite après reset si Toribash accepte run_frames.
    if AUTORUN and run_frames then
        pcall(function() run_frames(1) end)
    elseif AUTORUN and step_game then
        pcall(function() step_game() end)
    end
end)

add_hook("enter_frame", NAME .. "_frame", function()
    if not loaded then return end

    update_score_track()

    local action_frame = math.floor(tick / TURNFRAMES) * TURNFRAMES
    local pairs = action_by_frame[action_frame]
    if pairs then
        last_action_frame = action_frame
        apply_pairs(pairs)
        balance_pilot(action_frame)
    else
        last_pairs = 0
    end

    tick = tick + 1

    if tick >= MAX_TICKS then
        write_result("max_ticks")
    end

    if AUTORUN and tick < MAX_TICKS then
        if run_frames then
            pcall(function() run_frames(1) end)
        elseif step_game then
            pcall(function() step_game() end)
        end
    end
end)

add_hook("draw2d", NAME .. "_draw", function()
    if draw_text then
        draw_text("Xioi Master Final V3", 40, 60, 1)
        draw_text("loaded=" .. tostring(loaded) .. " err=" .. tostring(load_error), 40, 80, 1)
        local gen = agent and agent.generation or -1
        local cand = agent and agent.candidate or -1
        local pop = agent and agent.population or -1
        local gens = agent and agent.generations or -1
        draw_text("gen=" .. tostring(gen) .. "/" .. tostring(gens) .. " cand=" .. tostring(cand) .. "/" .. tostring(pop), 40, 100, 1)
        draw_text("tick=" .. tostring(tick) .. " action=" .. tostring(last_action_frame) .. " pairs=" .. tostring(last_pairs), 40, 120, 1)
        draw_text("ok=" .. tostring(motor_ok) .. " fail=" .. tostring(motor_fail) .. " method=" .. tostring(last_method), 40, 140, 1)
        local dy = 0
        if start_y and last_y then dy = last_y - start_y end
        draw_text("Y torso dy=" .. string.format("%.3f", dy) .. " correct_from=" .. tostring(CORRECT_FROM_FRAME), 40, 160, 1)
    end
end)
