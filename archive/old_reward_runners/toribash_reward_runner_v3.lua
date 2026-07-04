echo("################################################")
echo("[ToribashAI V3] LUA LOADED")
echo("################################################")

local CONFIG = {
    target_x = 0.0,
    target_y = -65.0,
    target_z = 5.4,
    max_frames = 300,
}

local frame = 0
local boot_ticks = 0
local running = false
local started_physics = false

local start_x = nil
local start_y = nil
local start_z = nil

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
                                -- Agent test v3 : pattern jambes alterné.
                                if frame % 80 < 40 then
                                    set_joint_state(0, 14, 1, true)
                                    set_joint_state(0, 15, 4, true)
                                    set_joint_state(0, 16, 2, true)

                                    set_joint_state(0, 17, 4, true)
                                    set_joint_state(0, 18, 1, true)
                                    set_joint_state(0, 19, 2, true)
                                    else
                                        set_joint_state(0, 14, 4, true)
                                        set_joint_state(0, 15, 1, true)
                                        set_joint_state(0, 16, 2, true)

                                        set_joint_state(0, 17, 1, true)
                                        set_joint_state(0, 18, 4, true)
                                        set_joint_state(0, 19, 2, true)
                                        end
                                        end

                                        local function start_physics_once()
                                        if started_physics then
                                            return
                                            end

                                            started_physics = true

                                            echo("[ToribashAI V3] STARTING PHYSICS")

                                            unfreeze_game()
                                            toggle_game_pause(false)
                                            run_frames(1)
                                            end

                                            local function finish_run()
                                            running = false

                                            local x, y, z = get_tori_center()

                                            if not x then
                                                echo("================================================")
                                                echo("[ToribashAI V3] NIL POSITION ERROR")
                                                echo("[ToribashAI V3] EVALUATION COMPLETE")
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
                                                    if z < 3.0 then
                                                        fell = true
                                                        end

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
                                                                        echo(string.format("[ToribashAI V3] FINAL SCORE = %.2f", score))
                                                                        echo(string.format("[ToribashAI V3] FINAL POS = %.2f %.2f %.2f", x, y, z))
                                                                        echo(string.format("[ToribashAI V3] FINAL DIST = %.2f", final_dist))
                                                                        echo(string.format("[ToribashAI V3] PROGRESS_Y = %.2f", progress_y))
                                                                        echo("[ToribashAI V3] FELL = " .. tostring(fell))
                                                                        echo("[ToribashAI V3] THIS IS V3")
                                                                        echo("[ToribashAI V3] EVALUATION COMPLETE")
                                                                        echo("================================================")

                                                                        freeze_game()
                                                                        end

                                                                        local function on_new_game()
                                                                        echo("[ToribashAI V3] NEW GAME")

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

                                                                                        if frame < 20 then
                                                                                            hold_all()
                                                                                            else
                                                                                                apply_agent_action()
                                                                                                end

                                                                                                run_frames(1)

                                                                                                if frame % 30 == 0 and x then
                                                                                                    local dist = distance_to_target(x, y, z)
                                                                                                    echo(string.format("[ToribashAI V3] frame=%d pos=%.2f %.2f %.2f dist=%.2f", frame, x, y, z, dist))
                                                                                                    end

                                                                                                    if frame >= CONFIG.max_frames then
                                                                                                        finish_run()
                                                                                                        end
                                                                                                        end

                                                                                                        remove_hooks("toribashai_reward_runner")
                                                                                                        remove_hooks("toribashai_reward_runner_v2")
                                                                                                        remove_hooks("toribashai_reward_runner_v3")

                                                                                                        add_hook("new_game", "toribashai_reward_runner_v3", on_new_game)
                                                                                                        add_hook("draw2d", "toribashai_reward_runner_v3", on_draw2d)
                                                                                                        add_hook("enter_frame", "toribashai_reward_runner_v3", on_enter_frame)
