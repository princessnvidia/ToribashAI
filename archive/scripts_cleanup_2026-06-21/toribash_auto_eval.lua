-- toribash_auto_eval.lua
-- Version sans os.execute, compatible avec le sandbox Lua de Toribash.
-- Lance avec : /lua toribash_auto_eval.lua

local CONFIG = {
    prefix = "ToribashAI_run_legs_gru_",
    count = 20,

    goal_x = 0.0,
    goal_y = -65.0,
    goal_z = 5.4,

    eval_frames = 700,

    results_path = "/home/vio/Documents/ToribashAI/evolution/results.jsonl",
    done_path = "/home/vio/Documents/ToribashAI/evolution/eval_done.txt",
}

local state = {
    index = 0,
    frame = 0,
    start_pos = nil,
    last_pos = nil,
    results_file = nil,
    running = false,
}

local function pad3(n)
if n < 10 then
    return "00" .. tostring(n)
    elseif n < 100 then
        return "0" .. tostring(n)
        end
        return tostring(n)
        end

        local function replay_name(i)
        return CONFIG.prefix .. pad3(i) .. ".rpl"
        end

        local function dist_to_goal(pos)
        local dx = pos.x - CONFIG.goal_x
        local dy = pos.y - CONFIG.goal_y
        local dz = pos.z - CONFIG.goal_z
        return math.sqrt(dx * dx + dy * dy + dz * dz)
        end

        local function json_escape(s)
        s = tostring(s)
        s = s:gsub("\\", "\\\\")
        s = s:gsub('"', '\\"')
        return s
        end

        local function get_tori_pos()
        if type(get_body_info) ~= "function" then
            return nil
            end

            -- Joueur 0 uniquement : Tori.
            -- On moyenne plusieurs body parts pour approximer le centre du corps.
            local ids = {0, 1, 2, 3, 4, 5}
            local sx, sy, sz, c = 0, 0, 0, 0

            for _, id in ipairs(ids) do
                local ok, body = pcall(get_body_info, 0, id)

                if ok and body then
                    local x, y, z = nil, nil, nil

                    if body.pos then
                        x = body.pos.x or body.pos[1]
                        y = body.pos.y or body.pos[2]
                        z = body.pos.z or body.pos[3]
                        elseif body.x and body.y and body.z then
                            x = body.x
                            y = body.y
                            z = body.z
                            elseif body[1] and body[2] and body[3] then
                                x = body[1]
                                y = body[2]
                                z = body[3]
                                end

                                if x and y and z then
                                    sx = sx + x
                                    sy = sy + y
                                    sz = sz + z
                                    c = c + 1
                                    end
                                    end
                                    end

                                    if c == 0 then
                                        return nil
                                        end

                                        return {
                                            x = sx / c,
                                            y = sy / c,
                                            z = sz / c,
                                        }
                                        end

                                        local function score_candidate(start_pos, end_pos)
                                        local start_dist = dist_to_goal(start_pos)
                                        local end_dist = dist_to_goal(end_pos)

                                        local progress_to_goal = start_dist - end_dist

                                        -- Cible en Y négatif, donc avancer = diminuer Y.
                                        local forward_y = start_pos.y - end_pos.y

                                        local score = 0
                                        score = score + progress_to_goal * 10.0
                                        score = score + forward_y * 2.0
                                        score = score - end_dist * 1.5

                                        -- Malus chute probable.
                                        if end_pos.z < 3.0 then
                                            score = score - 100.0
                                            end

                                            -- Bonus cible.
                                            if end_dist < 3.0 then
                                                score = score + 500.0
                                                elseif end_dist < 6.0 then
                                                    score = score + 200.0
                                                    elseif end_dist < 10.0 then
                                                        score = score + 80.0
                                                        end

                                                        return score, start_dist, end_dist, progress_to_goal, forward_y
                                                        end

                                                        local function write_error_result(name, err)
                                                        if not state.results_file then
                                                            return
                                                            end

                                                            local line = string.format(
                                                                '{"index":%d,"replay":"%s","score":-999999,"error":"%s"}\n',
                                                                state.index,
                                                                json_escape(name),
                                                                                       json_escape(err)
                                                            )

                                                            state.results_file:write(line)
                                                            state.results_file:flush()
                                                            end

                                                            local function write_result(name, score, start_dist, end_dist, progress, forward_y, pos)
                                                            if not state.results_file then
                                                                return
                                                                end

                                                                local line = string.format(
                                                                    '{"index":%d,"replay":"%s","score":%.6f,"start_distance":%.6f,"end_distance":%.6f,"progress":%.6f,"forward_y":%.6f,"final_x":%.6f,"final_y":%.6f,"final_z":%.6f}\n',
                                                                    state.index,
                                                                    json_escape(name),
                                                                                           score,
                                                                                           start_dist,
                                                                                           end_dist,
                                                                                           progress,
                                                                                           forward_y,
                                                                                               pos.x,
                                                                                           pos.y,
                                                                                           pos.z
                                                                )

                                                                state.results_file:write(line)
                                                                state.results_file:flush()
                                                                end

                                                                local function load_current()
                                                                local name = replay_name(state.index)

                                                                echo("[ToribashAI] Loading " .. name)

                                                                state.frame = 0
                                                                state.start_pos = nil
                                                                state.last_pos = nil
                                                                state.running = true

                                                                open_replay(name, true)

                                                                pcall(set_replay_speed, 1, true)
                                                                pcall(set_replay_speed, 1, false)
                                                                end

                                                                    local f = io.open(CONFIG.done_path, "w")
                                                                    if f then
                                                                        f:write("done\n")
                                                                        f:close()
                                                                        end

                                                                        remove_hooks("toribashai_auto_eval")
                                                                        end

                                                                        local function finish_current()
                                                                        local name = replay_name(state.index)

                                                                        if not state.start_pos or not state.last_pos then
                                                                            echo("[ToribashAI] No position data for " .. name)
                                                                            write_error_result(name, "no_position_data")
                                                                            else
                                                                                local score, start_dist, end_dist, progress, forward_y =
                                                                                score_candidate(state.start_pos, state.last_pos)

                                                                                echo(
                                                                                    string.format(
                                                                                        "[ToribashAI] %s score %.2f | dist %.2f -> %.2f | forward_y %.2f",
                                                                                        name,
                                                                                        score,
                                                                                        start_dist,
                                                                                        end_dist,
                                                                                        forward_y
                                                                                    )
                                                                                )

                                                                                write_result(name, score, start_dist, end_dist, progress, forward_y, state.last_pos)
                                                                                end

                                                                                state.index = state.index + 1

                                                                                if state.index >= CONFIG.count then
                                                                                    finish_all()
                                                                                    else
                                                                                        load_current()
                                                                                        end
                                                                                        end

                                                                                        local function on_frame()
                                                                                        if not state.running then
                                                                                            return
                                                                                            end

                                                                                            state.frame = state.frame + 1

                                                                                            local pos = get_tori_pos()

                                                                                            if pos then
                                                                                                state.last_pos = pos

                                                                                                if not state.start_pos and state.frame > 5 then
                                                                                                    state.start_pos = {
                                                                                                        x = pos.x,
                                                                                                        y = pos.y,
                                                                                                        z = pos.z,
                                                                                                    }
                                                                                                    end
                                                                                                    end

                                                                                                    if state.frame >= CONFIG.eval_frames then
                                                                                                        finish_current()
                                                                                                        end
                                                                                                        end

                                                                                                        local function start()
                                                                                                        state.results_file = io.open(CONFIG.results_path, "w")

                                                                                                        if not state.results_file then
                                                                                                            echo("[ToribashAI] ERROR: impossible d'ouvrir results.jsonl")
                                                                                                            return
                                                                                                            end

                                                                                                            local f = io.open(CONFIG.done_path, "w")
                                                                                                            if f then
                                                                                                                f:write("running\n")
                                                                                                                f:close()
                                                                                                                end

                                                                                                                state.index = 0

                                                                                                                remove_hooks("toribashai_auto_eval")
                                                                                                                add_hook("enter_frame", "toribashai_auto_eval", on_frame)

                                                                                                                echo("[ToribashAI] Auto eval started.")
                                                                                                                load_current()
                                                                                                                end

                                                                                                                echo("TORIBASH AI AUTOLOAD OK")
                                                                                                                start()


