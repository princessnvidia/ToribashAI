echo("################################################")
echo("[ToribashAI V6] LUA LOADED - AGENT FILE MODE")
echo("################################################")

dofile("toribashai_agent_current.lua")

local CONFIG = {
    target_x = 0.0,
    target_y = -65.0,
    target_z = 5.4,
    max_frames = 900,
    frames_per_action = 20,
}

local LEG_JOINTS = {14, 15, 16, 17, 18, 19}

local frame = 0
local boot_ticks = 0
local running = false
local started_physics = false

local start_x = nil
local start_y = nil
local start_z = nil

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

                        if c == 0 then return nil end
                            return sx / c, sy / c, sz / c
                            end

                            local function distance_to_target(x, y, z)
                            local dx = x - CONFIG.target_x
                            local dy = y - CONFIG.target_y
                            local dz = z - CONFIG.target_z
                            return math.sqrt(dx * dx + dy * dy + dz * dz)
                            end

                            local function hold_all()
                            for j = 0, 19 do
                                set_joint_state(0, j, 3, true)
                                end
                                end

                                local function apply_agent_action()
                                if not TORIBASHAI_AGENT or not TORIBASHAI_AGENT.actions then
                                    hold_all()
                                    return
                                    end

                                    local action_index = math.floor(frame / CONFIG.frames_per_action) + 1
                                    local action = TORIBASHAI_AGENT.actions[action_index]

                                    if not action then
                                        hold_all()
                                        return
                                        end

                                        for i, joint_id in ipairs(LEG_JOINTS) do
                                            local value = action[i] or 3
                                            set_joint_state(0, joint_id, value, true)
                                            end

                                            -- Corps stabilisé.
                                            set_joint_state(0, 0, 3, true)
                                            set_joint_state(0, 1, 3, true)
                                            set_joint_state(0, 2, 3, true)
                                            set_joint_state(0, 3, 1, true)
                                            set_joint_state(0, 12, 3, true)
                                            set_joint_state(0, 13, 3, true)
                                            end

                                            local function start_physics_once()
                                            if started_physics then return end

                                                started_physics = true
                                                echo("[ToribashAI V6] STARTING PHYSICS")

                                                unfreeze_game()
                                                toggle_game_pause(false)
                                                step_game(false, false)
                                                run_frames(1)
                                                run_frames(10)
                                                end

                                                local function finish_run()
                                                running = false

                                                local x, y, z = get_tori_center()

                                                if not x then
                                                    echo("================================================")
                                                    echo("[ToribashAI V6] NIL POSITION ERROR")
                                                    echo("[ToribashAI V6] EVALUATION COMPLETE")
                                                    echo("================================================")
                                                    freeze_game()
                                                    return
                                                    end

                                                    local final_dist = distance_to_target(x, y, z)

                                                    local progress_y = 0.0
                                                    if start_y then
                                                        progress_y = start_y - y
                                                        end

                                                        local fell = false
                                                        if z < 3.0 then fell = true end

                                                            local score = 0.0
                                                            score = score + progress_y * 10.0
                                                            score = score - final_dist * 2.0

                                                            if fell then
                                                                score = score - 150.0
                                                                end

                                                                if final_dist < 3.0 then
                                                                    score = score + 1000.0
                                                                    elseif final_dist < 6.0 then
                                                                        score = score + 500.0
                                                                        elseif final_dist < 10.0 then
                                                                            score = score + 200.0
                                                                            end

                                                                            echo("================================================")
                                                                            echo("[ToribashAI V6] AGENT = " .. tostring(TORIBASHAI_AGENT.name))
                                                                            echo(string.format("[ToribashAI V6] FINAL SCORE = %.2f", score))
                                                                            echo(string.format("[ToribashAI V6] FINAL POS = %.2f %.2f %.2f", x, y, z))
                                                                            echo(string.format("[ToribashAI V6] FINAL DIST = %.2f", final_dist))
                                                                            echo(string.format("[ToribashAI V6] PROGRESS_Y = %.2f", progress_y))
                                                                            echo("[ToribashAI V6] FELL = " .. tostring(fell))
                                                                            echo("[ToribashAI V6] THIS IS V6")
                                                                            echo("[ToribashAI V6] EVALUATION COMPLETE")
                                                                            echo("================================================")

                                                                            freeze_game()
                                                                            end

                                                                            local function on_new_game()
                                                                            echo("[ToribashAI V6] NEW GAME")
                                                                            echo("[ToribashAI V6] LOADED AGENT = " .. tostring(TORIBASHAI_AGENT.name))

                                                                            frame = 0
                                                                            boot_ticks = 0
                                                                            running = true
                                                                            started_physics = false

                                                                            start_x = nil
                                                                            start_y = nil
                                                                            start_z = nil

                                                                            hold_all()
                                                                            unfreeze_game()
                                                                            end

                                                                            local function on_draw2d()
                                                                            if not running or started_physics then return end

                                                                                boot_ticks = boot_ticks + 1
                                                                                if boot_ticks >= 30 then
                                                                                    start_physics_once()
                                                                                    end
                                                                                    end

                                                                                    local function on_enter_frame()
                                                                                    if not running then return end

                                                                                        frame = frame + 1

                                                                                        local x, y, z = get_tori_center()

                                                                                        if x and not start_x then
                                                                                            start_x = x
                                                                                            start_y = y
                                                                                            start_z = z
                                                                                            end

                                                                                            if frame < 20 then
                                                                                                hold_all()
                                                                                                else
                                                                                                    apply_agent_action()
                                                                                                    end

                                                                                                    run_frames(1)

                                                                                                    if frame % 60 == 0 and x then
                                                                                                        local dist = distance_to_target(x, y, z)
                                                                                                        local action_index = math.floor(frame / CONFIG.frames_per_action) + 1
                                                                                                        echo(string.format(
                                                                                                            "[ToribashAI V6] frame=%d action=%d pos=%.2f %.2f %.2f dist=%.2f",
                                                                                                            frame, action_index, x, y, z, dist
                                                                                                        ))
                                                                                                        end

                                                                                                        if frame >= CONFIG.max_frames then
                                                                                                            finish_run()
                                                                                                            end
                                                                                                            end

                                                                                                            remove_hooks("toribashai_reward_runner")
                                                                                                            remove_hooks("toribashai_reward_runner_v2")
                                                                                                            remove_hooks("toribashai_reward_runner_v3")
                                                                                                            remove_hooks("toribashai_reward_runner_v4")
                                                                                                            remove_hooks("toribashai_reward_runner_v5")
                                                                                                            remove_hooks("toribashai_reward_runner_v6")

                                                                                                            add_hook("new_game", "toribashai_reward_runner_v6", on_new_game)
                                                                                                            add_hook("draw2d", "toribashai_reward_runner_v6", on_draw2d)
                                                                                                            add_hook("enter_frame", "toribashai_reward_runner_v6", on_enter_frame)
