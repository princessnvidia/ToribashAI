echo("################################################")
echo("[ToribashAI Balance V2] LUA LOADED - 500 FRAMES")
echo("################################################")

local CONFIG = {
    max_frames = 500,
    warmup_frames = 20,

    -- Plus strict que V1 : on élimine les postures trop basses plus vite.
    fall_z = 4.5,

    result_path = "toribashai_episode_result.json"
}

local frame = 0
local boot_ticks = 0
local running = false
local started_physics = false
local finished = false

local start_x = nil
local start_y = nil
local start_z = nil

local AGENT_JOINTS = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9,
    10, 11, 12, 13,
    14, 15, 16, 17, 18, 19
}

local function load_agent()
TORIBASHAI_AGENT = nil

local ok, err = pcall(function()
dofile("toribashai_agent_current.lua")
end)

if not ok then
    echo("[Balance V2] AGENT LOAD ERROR")
    echo(tostring(err))
    return false
    end

    echo("################################################")
    echo("[Balance V2] AGENT RELOADED")
    echo("[Balance V2] agent = " .. tostring(TORIBASHAI_AGENT.name))

    if TORIBASHAI_AGENT.actions then
        echo("[Balance V2] actions = " .. tostring(#TORIBASHAI_AGENT.actions))
        else
            echo("[Balance V2] WARNING: no actions")
            end

            echo("################################################")

            return true
            end

            local function clamp_joint_value(value)
            value = tonumber(value) or 3

            if value < 1 then
                value = 3
                end

                if value > 4 then
                    value = 4
                    end

                    return value
                    end

                    local function hold_all()
                    for j = 0, 19 do
                        set_joint_state(0, j, 3, true)
                        end
                        end

                        local function body_pos(player, body)
                        local info = get_body_info(player, body)

                        if not info then
                            return nil
                            end

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

                                        local function get_tori_center()
                                        local ids = {0, 1, 2, 3, 4, 5}
                                        local sx, sy, sz, c = 0, 0, 0, 0

                                        for _, id in ipairs(ids) do
                                            local ok, x, y, z = pcall(body_pos, 0, id)

                                            if ok and x and y and z then
                                                sx = sx + x
                                                sy = sy + y
                                                sz = sz + z
                                                c = c + 1
                                                end
                                                end

                                                if c == 0 then
                                                    return nil
                                                    end

                                                    return sx / c, sy / c, sz / c
                                                    end

                                                    local function get_posture_z(fallback_z)
                                                    local hx, hy, hz = body_pos(0, 0)
                                                    local cx, cy, cz = body_pos(0, 1)
                                                    local lsx, lsy, lsz = body_pos(0, 11)
                                                    local rsx, rsy, rsz = body_pos(0, 12)

                                                    if hz and cz and lsz and rsz then
                                                        return hz * 0.45 + cz * 0.25 + lsz * 0.15 + rsz * 0.15
                                                        end

                                                        return fallback_z or 0
                                                        end

                                                        local function apply_agent_action()
                                                        if not TORIBASHAI_AGENT or not TORIBASHAI_AGENT.actions then
                                                            hold_all()
                                                            return
                                                            end

                                                            local action_index = math.floor((frame - CONFIG.warmup_frames) / 20) + 1

                                                            if action_index < 1 then
                                                                action_index = 1
                                                                end

                                                                local action = TORIBASHAI_AGENT.actions[action_index]

                                                                if not action then
                                                                    hold_all()
                                                                    return
                                                                    end

                                                                    local control_joints = TORIBASHAI_AGENT.control_joints or AGENT_JOINTS

                                                                    for i, joint_id in ipairs(control_joints) do
                                                                        local value = clamp_joint_value(action[i] or 3)
                                                                        set_joint_state(0, joint_id, value, true)
                                                                        end
                                                                        end

                                                                        local function write_result(score, x, y, z, posture_z, drift, fell, reason)
                                                                        local f = io.open(CONFIG.result_path, "w")

                                                                        if not f then
                                                                            echo("[Balance V2] ERROR: cannot write result file")
                                                                            return
                                                                            end

                                                                            f:write(string.format(
                                                                                '{"score": %.4f, "x": %.4f, "y": %.4f, "z": %.4f, "posture_z": %.4f, "drift": %.4f, "fell": %s, "frames": %d, "agent": "%s", "reason": "%s"}\n',
                                                                                score,
                                                                                x,
                                                                                y,
                                                                                z,
                                                                                posture_z,
                                                                                drift,
                                                                                tostring(fell),
                                                                                                  frame,
                                                                                                  tostring(TORIBASHAI_AGENT and TORIBASHAI_AGENT.name or "unknown"),
                                                                                                  tostring(reason)
                                                                            ))

                                                                            f:close()
                                                                            echo("[Balance V2] Result written: " .. CONFIG.result_path)
                                                                            end

                                                                            local function finish_run(reason)
                                                                            if finished then
                                                                                return
                                                                                end

                                                                                finished = true
                                                                                running = false

                                                                                local x, y, z = get_tori_center()

                                                                                if not x then
                                                                                    freeze_game()
                                                                                    return
                                                                                    end

                                                                                    local posture_z = get_posture_z(z)

                                                                                    local dx = x - (start_x or x)
                                                                                    local dy = y - (start_y or y)
                                                                                    local drift = math.sqrt(dx * dx + dy * dy)

                                                                                    local fell = z < CONFIG.fall_z

                                                                                    local score = 0

                                                                                    -- Objectif principal : rester haut.
                                                                                    score = score + posture_z * 35.0

                                                                                    -- Objectif secondaire : rester stable.
                                                                                    score = score - drift * 20.0

                                                                                    -- Survivre longtemps, plus fort que V1.
                                                                                    score = score + frame * 2.0

                                                                                    -- Ne pas tomber.
                                                                                    if fell then
                                                                                        score = score - 500.0
                                                                                        end

                                                                                        -- Pénalité forte si tête/épaules/torse bas.
                                                                                        if posture_z < 5.8 then
                                                                                            score = score - 300.0
                                                                                            end

                                                                                            -- Bonus si posture très haute.
                                                                                            if posture_z > 7.0 then
                                                                                                score = score + 150.0
                                                                                                end

                                                                                                echo("================================================")
                                                                                                echo("[Balance V2] SCORE = " .. tostring(score))
                                                                                                echo(string.format("[Balance V2] pos=%.2f %.2f %.2f posture=%.2f drift=%.2f", x, y, z, posture_z, drift))
                                                                                                echo("[Balance V2] fell=" .. tostring(fell) .. " reason=" .. tostring(reason))
                                                                                                echo("================================================")

                                                                                                write_result(score, x, y, z, posture_z, drift, fell, reason)
                                                                                                freeze_game()
                                                                                                end

                                                                                                local function start_physics_once()
                                                                                                if started_physics then
                                                                                                    return
                                                                                                    end

                                                                                                    started_physics = true

                                                                                                    echo("[Balance V2] STARTING PHYSICS")

                                                                                                    unfreeze_game()
                                                                                                    toggle_game_pause(false)
                                                                                                    step_game(false, false)
                                                                                                    run_frames(1)
                                                                                                    run_frames(10)
                                                                                                    end

                                                                                                    local function on_new_game()
                                                                                                    local ok = load_agent()

                                                                                                    if not ok then
                                                                                                        running = false
                                                                                                        freeze_game()
                                                                                                        return
                                                                                                        end

                                                                                                        echo("[Balance V2] NEW GAME")

                                                                                                        frame = 0
                                                                                                        boot_ticks = 0
                                                                                                        running = true
                                                                                                        started_physics = false
                                                                                                        finished = false

                                                                                                        start_x = nil
                                                                                                        start_y = nil
                                                                                                        start_z = nil

                                                                                                        hold_all()
                                                                                                        unfreeze_game()
                                                                                                        end

                                                                                                        local function on_draw2d()
                                                                                                        if not running or started_physics then
                                                                                                            return
                                                                                                            end

                                                                                                            boot_ticks = boot_ticks + 1

                                                                                                            if boot_ticks >= 30 then
                                                                                                                start_physics_once()
                                                                                                                end
                                                                                                                end

                                                                                                                local function on_enter_frame()
                                                                                                                if not running then
                                                                                                                    return
                                                                                                                    end

                                                                                                                    frame = frame + 1

                                                                                                                    local x, y, z = get_tori_center()

                                                                                                                    if x and not start_x then
                                                                                                                        start_x = x
                                                                                                                        start_y = y
                                                                                                                        start_z = z
                                                                                                                        end

                                                                                                                        if frame < CONFIG.warmup_frames then
                                                                                                                            hold_all()
                                                                                                                            else
                                                                                                                                apply_agent_action()
                                                                                                                                apply_agent_action()
                                                                                                                                end

                                                                                                                                run_frames(1)

                                                                                                                                if frame % 100 == 0 and x then
                                                                                                                                    echo(string.format(
                                                                                                                                        "[Balance V2] frame=%d pos=%.2f %.2f %.2f posture=%.2f",
                                                                                                                                        frame,
                                                                                                                                        x,
                                                                                                                                        y,
                                                                                                                                        z,
                                                                                                                                        get_posture_z(z)
                                                                                                                                    ))
                                                                                                                                    end

                                                                                                                                    if frame >= CONFIG.max_frames then
                                                                                                                                        finish_run("max_frames")
                                                                                                                                        return
                                                                                                                                        end

                                                                                                                                        if x and z < CONFIG.fall_z then
                                                                                                                                            finish_run("fell")
                                                                                                                                            return
                                                                                                                                            end
                                                                                                                                            end

                                                                                                                                            remove_hooks("toribashai_balance_runner_v1")
                                                                                                                                            remove_hooks("toribashai_balance_runner_v2")

                                                                                                                                            add_hook("new_game", "toribashai_balance_runner_v2", on_new_game)
                                                                                                                                            add_hook("draw2d", "toribashai_balance_runner_v2", on_draw2d)
                                                                                                                                            add_hook("enter_frame", "toribashai_balance_runner_v2", on_enter_frame)
