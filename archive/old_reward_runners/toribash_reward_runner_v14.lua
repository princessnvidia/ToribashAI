echo("################################################")
echo("[ToribashAI V15] LUA LOADED - FILE RESET REQUEST")
echo("################################################")

local CONFIG = {
    target_x = 0.0,
    target_y = -65.0,
    target_z = 5.4,

    max_frames = 300,
    frames_per_action = 20,
    warmup_frames = 20,

    early_stop_frames = 120,
    min_progress_y = -0.5,
    fall_z = 3.0,

    result_path = "toribashai_episode_result.json",
    reset_request_path = "toribashai_reset_request.txt"
}

local AGENT_JOINTS = {4, 5, 6, 7, 14, 15, 16, 17, 18, 19}

local frame = 0
local boot_ticks = 0
local running = false
local started_physics = false
local start_y = nil
local finished = false

local function load_agent()
TORIBASHAI_AGENT = nil

local ok, err = pcall(function()
dofile("toribashai_agent_current.lua")
end)

if not ok then
    echo("[ToribashAI V15] AGENT LOAD ERROR")
    echo(tostring(err))
    return false
    end

    echo("################################################")
    echo("[ToribashAI V15] AGENT RELOADED")
    echo("[ToribashAI V15] loaded agent = " .. tostring(TORIBASHAI_AGENT.name))
    echo("################################################")

    return true
    end

    local function hold_all()
    for j = 0, 19 do
        set_joint_state(0, j, 3, true)
        end
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

                                    local function dist_to_target(x, y, z)
                                    local dx = x - CONFIG.target_x
                                    local dy = y - CONFIG.target_y
                                    local dz = z - CONFIG.target_z
                                    return math.sqrt(dx * dx + dy * dy + dz * dz)
                                    end

                                    local function write_result(score, x, y, z, final_dist, progress_y, fell, reason)
                                    local f = io.open(CONFIG.result_path, "w")

                                    if not f then
                                        echo("[ToribashAI V15] ERROR: cannot write result file")
                                        echo("[ToribashAI V15] path = " .. CONFIG.result_path)
                                        return
                                        end

                                        f:write(string.format(
                                            '{"score": %.4f, "x": %.4f, "y": %.4f, "z": %.4f, "final_dist": %.4f, "progress_y": %.4f, "fell": %s, "frames": %d, "agent": "%s", "reason": "%s"}\n',
                                            score,
                                            x,
                                            y,
                                            z,
                                            final_dist,
                                            progress_y,
                                            tostring(fell),
                                                              frame,
                                                              tostring(TORIBASHAI_AGENT.name),
                                                              tostring(reason)
                                        ))

                                        f:close()
                                        echo("[ToribashAI V15] Result written: " .. CONFIG.result_path)
                                        end

                                        local function apply_extra_arms()
                                        if frame % 80 < 40 then
                                            set_joint_state(0, 8, 1, true)
                                            set_joint_state(0, 9, 4, true)
                                            else
                                                set_joint_state(0, 8, 4, true)
                                                set_joint_state(0, 9, 1, true)
                                                end
                                                end

                                                local function apply_agent_action()
                                                if not TORIBASHAI_AGENT or not TORIBASHAI_AGENT.actions then
                                                    echo("[ToribashAI V15] ERROR agent/actions nil")
                                                    hold_all()
                                                    return
                                                    end

                                                    local action_index = math.floor((frame - CONFIG.warmup_frames) / CONFIG.frames_per_action) + 1
                                                    if action_index < 1 then action_index = 1 end

                                                        local action = TORIBASHAI_AGENT.actions[action_index]

                                                        if not action then
                                                            hold_all()
                                                            return
                                                            end

                                                            for i, joint_id in ipairs(AGENT_JOINTS) do
                                                                local value = action[i] or 3
                                                                set_joint_state(0, joint_id, value, true)
                                                                end

                                                                apply_extra_arms()

                                                                set_joint_state(0, 0, 3, true)
                                                                set_joint_state(0, 1, 3, true)
                                                                set_joint_state(0, 2, 3, true)
                                                                set_joint_state(0, 3, 1, true)
                                                                set_joint_state(0, 10, 3, true)
                                                                set_joint_state(0, 11, 3, true)
                                                                set_joint_state(0, 12, 3, true)
                                                                set_joint_state(0, 13, 3, true)
                                                                end

                                                                local function start_physics_once()
                                                                if started_physics then return end

                                                                    started_physics = true
                                                                    echo("[ToribashAI V15] STARTING PHYSICS")

                                                                    unfreeze_game()
                                                                    toggle_game_pause(false)
                                                                    step_game(false, false)
                                                                    run_frames(1)
                                                                    run_frames(10)
                                                                    end

                                                                    local function finish_run(reason)
                                                                    if finished then return end

                                                                        finished = true
                                                                        running = false

                                                                        local x, y, z = get_tori_center()
                                                                        if not x then
                                                                            echo("[ToribashAI V15] NIL POSITION ERROR")
                                                                            freeze_game()
                                                                            return
                                                                            end

                                                                            local final_dist = dist_to_target(x, y, z)
                                                                            local progress_y = 0.0
                                                                            if start_y then progress_y = start_y - y end

                                                                                local fell = z < CONFIG.fall_z
                                                                                local score = progress_y * 10.0 - final_dist * 2.0

                                                                                if fell then score = score - 150.0 end
                                                                                    if reason == "bad_progress" then score = score - 25.0 end

                                                                                        echo("================================================")
                                                                                        echo("[ToribashAI V15] AGENT = " .. tostring(TORIBASHAI_AGENT.name))
                                                                                        echo(string.format("[ToribashAI V15] FINAL SCORE = %.2f", score))
                                                                                        echo(string.format("[ToribashAI V15] FINAL POS = %.2f %.2f %.2f", x, y, z))
                                                                                        echo(string.format("[ToribashAI V15] FINAL DIST = %.2f", final_dist))
                                                                                        echo(string.format("[ToribashAI V15] PROGRESS_Y = %.2f", progress_y))
                                                                                        echo("[ToribashAI V15] FELL = " .. tostring(fell))
                                                                                        echo("[ToribashAI V15] REASON = " .. tostring(reason))
                                                                                        echo("================================================")

                                                                                        write_result(score, x, y, z, final_dist, progress_y, fell, reason)
                                                                                        freeze_game()
                                                                                        end

                                                                                        local function check_reset_request()
                                                                                        if running then return end

                                                                                            local f = io.open(CONFIG.reset_request_path, "r")
                                                                                            if not f then return end

                                                                                                f:close()
                                                                                                os.remove(CONFIG.reset_request_path)

                                                                                                echo("[ToribashAI V15] RESET REQUEST DETECTED")
                                                                                                runCmd("reset")
                                                                                                end

                                                                                                local function on_new_game()
                                                                                                local ok = load_agent()
                                                                                                if not ok then
                                                                                                    running = false
                                                                                                    freeze_game()
                                                                                                    return
                                                                                                    end

                                                                                                    echo("[ToribashAI V15] NEW GAME")

                                                                                                    frame = 0
                                                                                                    boot_ticks = 0
                                                                                                    running = true
                                                                                                    started_physics = false
                                                                                                    start_y = nil
                                                                                                    finished = false

                                                                                                    hold_all()
                                                                                                    unfreeze_game()
                                                                                                    end

                                                                                                    local function on_draw2d()
                                                                                                    check_reset_request()

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
                                                                                                                if x and not start_y then start_y = y end

                                                                                                                    if frame < CONFIG.warmup_frames then
                                                                                                                        hold_all()
                                                                                                                        else
                                                                                                                            apply_agent_action()
                                                                                                                            apply_agent_action()
                                                                                                                            end

                                                                                                                            run_frames(1)

                                                                                                                            if frame >= CONFIG.warmup_frames then
                                                                                                                                apply_agent_action()
                                                                                                                                end

                                                                                                                                if frame % 60 == 0 and x then
                                                                                                                                    echo(string.format(
                                                                                                                                        "[ToribashAI V15] frame=%d pos=%.2f %.2f %.2f dist=%.2f",
                                                                                                                                        frame, x, y, z, dist_to_target(x, y, z)
                                                                                                                                    ))
                                                                                                                                    end

                                                                                                                                    if frame >= CONFIG.early_stop_frames and x and start_y then
                                                                                                                                        local progress_y = start_y - y

                                                                                                                                        if z < CONFIG.fall_z then
                                                                                                                                            echo("[ToribashAI V15] EARLY STOP: fell")
                                                                                                                                            finish_run("fell")
                                                                                                                                            return
                                                                                                                                            end

                                                                                                                                            if progress_y < CONFIG.min_progress_y then
                                                                                                                                                echo("[ToribashAI V15] EARLY STOP: bad_progress")
                                                                                                                                                finish_run("bad_progress")
                                                                                                                                                return
                                                                                                                                                end
                                                                                                                                                end

                                                                                                                                                if frame >= CONFIG.max_frames then
                                                                                                                                                    finish_run("max_frames")
                                                                                                                                                    return
                                                                                                                                                    end
                                                                                                                                                    end

                                                                                                                                                    remove_hooks("toribashai_reward_runner")
                                                                                                                                                    remove_hooks("toribashai_reward_runner_v2")
                                                                                                                                                    remove_hooks("toribashai_reward_runner_v3")
                                                                                                                                                    remove_hooks("toribashai_reward_runner_v4")
                                                                                                                                                    remove_hooks("toribashai_reward_runner_v5")
                                                                                                                                                    remove_hooks("toribashai_reward_runner_v6")
                                                                                                                                                    remove_hooks("toribashai_reward_runner_v7")
                                                                                                                                                    remove_hooks("toribashai_reward_runner_v8")
                                                                                                                                                    remove_hooks("toribashai_reward_runner_v9")
                                                                                                                                                    remove_hooks("toribashai_reward_runner_v10")
                                                                                                                                                    remove_hooks("toribashai_reward_runner_v11")
                                                                                                                                                    remove_hooks("toribashai_reward_runner_v12")
                                                                                                                                                    remove_hooks("toribashai_reward_runner_v13")
                                                                                                                                                    remove_hooks("toribashai_reward_runner_V15")

                                                                                                                                                    add_hook("new_game", "toribashai_reward_runner_V15", on_new_game)
                                                                                                                                                    add_hook("draw2d", "toribashai_reward_runner_V15", on_draw2d)
                                                                                                                                                    add_hook("enter_frame", "toribashai_reward_runner_V15", on_enter_frame)
