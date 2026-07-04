-- toribash_reward_runner_v13_4.lua
-- V13.4 propre :
-- - support agent.commands exact RPL clone
-- - support agent.actions ancien format
-- - auto-run sans appuyer sur espace
-- - reset propre
-- - score + anti-plongeon

local result_path = "../script/toribashai_episode_result.json"
local agent_path = "toribashai_agent_current.lua"

local max_frames = 2000
local early_stop_frames = 999999

local agent = nil
local agent_loaded = false
local done = false

local command_index = 1
local action_index = 1

local start_x = 0
local start_y = 0
local start_z = 0
local last_progress_y = 0
local stagnant_frames = 0

local control_joints_default = {
    4, 5, 6, 7,
    14, 15, 16,
    17, 18, 19
}

local function echo_safe(msg)
if echo then
    echo(msg)
    end
    end

    local function auto_advance()
    if run_frames then
        run_frames(1)
        return
        end

        if step_game then
            step_game()
            return
            end
            end

            local function get_frame()
            local ws = get_world_state()
            if ws and ws.match_frame then
                return ws.match_frame
                end
                return 0
                end

                local function get_body_pos()
                local x, y, z = get_body_info(0, 0)
                return x or 0, y or 0, z or 0
                end

                local function safe_joint_state(joint, value)
                joint = tonumber(joint)
                value = tonumber(value)

                if joint == nil or value == nil then
                    return
                    end

                    if value < 1 then value = 3 end
                        if value > 4 then value = 4 end

                            set_joint_state(0, joint, value)
                            end

                            local function write_result(score, x, y, z, final_dist, progress_y, fell, frames, reason)
                            local f = io.open(result_path, "w")
                            if not f then
                                return
                                end

                                f:write("{\n")
                                f:write(string.format('  "score": %.4f,\n', score))
                                f:write(string.format('  "x": %.4f,\n', x))
                                f:write(string.format('  "y": %.4f,\n', y))
                                f:write(string.format('  "z": %.4f,\n', z))
                                f:write(string.format('  "final_dist": %.4f,\n', final_dist))
                                f:write(string.format('  "progress_y": %.4f,\n', progress_y))
                                f:write(string.format('  "fell": %s,\n', tostring(fell)))
                                f:write(string.format('  "frames": %d,\n', frames))
                                f:write(string.format('  "agent": "%s",\n', agent and agent.name or "unknown"))
                                f:write(string.format('  "reason": "%s"\n', reason))
                                f:write("}\n")
                                f:close()
                                end

                                local function load_agent()
                                agent = dofile(agent_path)

                                if not agent then
                                    echo_safe("ToribashAI V13.4: impossible de charger l'agent")
                                    return false
                                    end

                                    command_index = 1
                                    action_index = 1
                                    agent_loaded = true

                                    echo_safe("ToribashAI V13.4 agent: " .. tostring(agent.name))

                                    if agent.commands then
                                        echo_safe("Mode: commands / RPL clone")
                                        echo_safe("Commands: " .. tostring(#agent.commands))
                                        elseif agent.actions then
                                            echo_safe("Mode: actions")
                                            echo_safe("Actions: " .. tostring(#agent.actions))
                                            else
                                                echo_safe("Agent invalide: pas de commands/actions")
                                                end

                                                return true
                                                end

                                                local function apply_commands(frame)
                                                if not agent or not agent.commands then
                                                    return
                                                    end

                                                    while command_index <= #agent.commands do
                                                        local cmd = agent.commands[command_index]

                                                        if not cmd or not cmd.frame then
                                                            command_index = command_index + 1
                                                            elseif cmd.frame > frame then
                                                                break
                                                                else
                                                                    if cmd.frame == frame and cmd.joints then
                                                                        for _, pair in ipairs(cmd.joints) do
                                                                            safe_joint_state(pair[1], pair[2])
                                                                            end
                                                                            end

                                                                            command_index = command_index + 1
                                                                            end
                                                                            end
                                                                            end

                                                                            local function get_turnframes()
                                                                            local ws = get_world_state()

                                                                            if ws and ws.turn_frame then
                                                                                return ws.turn_frame
                                                                                end

                                                                                return 10
                                                                                end

                                                                                local function apply_actions(frame)
                                                                                if not agent or not agent.actions then
                                                                                    return
                                                                                    end

                                                                                    local turnframes = get_turnframes()

                                                                                    if frame % turnframes ~= 0 then
                                                                                        return
                                                                                        end

                                                                                        local action = agent.actions[action_index]
                                                                                        if not action then
                                                                                            return
                                                                                            end

                                                                                            local control_joints = agent.control_joints or control_joints_default

                                                                                            for i, value in ipairs(action) do
                                                                                                local joint = control_joints[i]
                                                                                                if joint ~= nil then
                                                                                                    safe_joint_state(joint, value)
                                                                                                    end
                                                                                                    end

                                                                                                    action_index = action_index + 1
                                                                                                    end

                                                                                                    local function evaluate(frame, reason)
                                                                                                    local x, y, z = get_body_pos()

                                                                                                    local progress_y = y - start_y
                                                                                                    local dx = x - start_x

                                                                                                    local target_y = -65
                                                                                                    local final_dist = math.abs(target_y - y)

                                                                                                    local fell = false
                                                                                                    if z < 3.5 then
                                                                                                        fell = true
                                                                                                        end

                                                                                                        local score = 0
                                                                                                        score = score + progress_y * 12.0
                                                                                                        score = score - final_dist * 1.8
                                                                                                        score = score - math.abs(dx) * 1.2

                                                                                                        -- Anti-plongeon / anti-levier
                                                                                                        if progress_y > 2.5 and z < 5.6 then
                                                                                                            score = score - 300
                                                                                                            end

                                                                                                            if z < 5.2 then
                                                                                                                score = score - 80
                                                                                                                end

                                                                                                                if fell then
                                                                                                                    score = score - 250
                                                                                                                    end

                                                                                                                    write_result(score, x, y, z, final_dist, progress_y, fell, frame, reason)
                                                                                                                    done = true
                                                                                                                    end

                                                                                                                    local function on_enter_frame()
                                                                                                                    if done then
                                                                                                                        return
                                                                                                                        end

                                                                                                                        if not agent_loaded then
                                                                                                                            local ok = load_agent()
                                                                                                                            if not ok then
                                                                                                                                auto_advance()
                                                                                                                                return
                                                                                                                                end

                                                                                                                                start_x, start_y, start_z = get_body_pos()
                                                                                                                                last_progress_y = 0
                                                                                                                                stagnant_frames = 0
                                                                                                                                end

                                                                                                                                local frame = get_frame()

                                                                                                                                if agent.commands then
                                                                                                                                    apply_commands(frame)
                                                                                                                                    else
                                                                                                                                        apply_actions(frame)
                                                                                                                                        end

                                                                                                                                        local x, y, z = get_body_pos()
                                                                                                                                        local progress_y = y - start_y

                                                                                                                                        if math.abs(progress_y - last_progress_y) < 0.01 then
                                                                                                                                            stagnant_frames = stagnant_frames + 1
                                                                                                                                            else
                                                                                                                                                stagnant_frames = 0
                                                                                                                                                end

                                                                                                                                                last_progress_y = progress_y

                                                                                                                                                if frame >= max_frames then
                                                                                                                                                    evaluate(frame, "max_frames")
                                                                                                                                                    return
                                                                                                                                                    end

                                                                                                                                                    if stagnant_frames >= early_stop_frames then
                                                                                                                                                        evaluate(frame, "stagnant")
                                                                                                                                                        return
                                                                                                                                                        end

                                                                                                                                                        auto_advance()
                                                                                                                                                        end

                                                                                                                                                        local function on_new_game()
                                                                                                                                                        agent_loaded = false
                                                                                                                                                        done = false

                                                                                                                                                        command_index = 1
                                                                                                                                                        action_index = 1

                                                                                                                                                        start_x = 0
                                                                                                                                                        start_y = 0
                                                                                                                                                        start_z = 0

                                                                                                                                                        last_progress_y = 0
                                                                                                                                                        stagnant_frames = 0

                                                                                                                                                        os.remove(result_path)

                                                                                                                                                        echo_safe("ToribashAI V13.4 reset")
                                                                                                                                                        auto_advance()
                                                                                                                                                        end

                                                                                                                                                        add_hook("enter_frame", "toribashai_v13_4_enter_frame", on_enter_frame)
                                                                                                                                                        add_hook("new_game", "toribashai_v13_4_new_game", on_new_game)

                                                                                                                                                        echo_safe("ToribashAI reward runner V13.4 loaded")
