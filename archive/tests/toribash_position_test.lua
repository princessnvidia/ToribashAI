echo("[ToribashAI] POSITION TEST LOADED")

local frame = 0
local running = false
local started_physics = false

local function body_pos(player, body)
local info = get_body_info(player, body)

if info == nil then
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

                            local function hold_all()
                            for j = 0, 19 do
                                set_joint_state(0, j, 3, true)
                                end
                                end

                                local function apply_step()
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

                                            echo("[ToribashAI] Starting physics without space")

                                            unfreeze_game()

                                            -- Équivalent du premier espace si le jeu est encore en pause.
                                            if is_game_paused and is_game_paused() then
                                                toggle_game_pause(false)
                                                else
                                                    toggle_game_pause(false)
                                                    end

                                                    run_frames(1)
                                                    end

                                                    local function on_new_game()
                                                    echo("[ToribashAI] NEW GAME -> POSITION TEST")

                                                    frame = 0
                                                    running = true
                                                    started_physics = false

                                                    hold_all()
                                                    unfreeze_game()

                                                    -- Tente de remplacer le premier appui sur espace dès le chargement.
                                                    toggle_game_pause(false)
                                                    end

                                                    local function on_enter_frame()
                                                    if not running then
                                                        return
                                                        end

                                                        frame = frame + 1

                                                        if frame == 2 then
                                                            start_physics_once()
                                                            end

                                                            if frame < 20 then
                                                                hold_all()
                                                                else
                                                                    apply_step()
                                                                    end

                                                                    -- Force l’avancement de la simulation.
                                                                    run_frames(1)

                                                                    if frame % 30 == 0 then
                                                                        local x, y, z = get_tori_center()

                                                                        if x then
                                                                            echo(string.format("[ToribashAI] frame=%d pos=%.2f %.2f %.2f", frame, x, y, z))
                                                                            else
                                                                                echo("[ToribashAI] frame=" .. frame .. " pos=nil")
                                                                                end
                                                                                end

                                                                                if frame >= 300 then
                                                                                    running = false

                                                                                    local x, y, z = get_tori_center()
                                                                                    if x then
                                                                                        echo(string.format("[ToribashAI] FINAL pos=%.2f %.2f %.2f", x, y, z))
                                                                                        end

                                                                                        freeze_game()
                                                                                        echo("[ToribashAI] POSITION TEST DONE")
                                                                                        end
                                                                                        end

                                                                                        remove_hooks("toribashai_position_test")
                                                                                        add_hook("new_game", "toribashai_position_test", on_new_game)
                                                                                        add_hook("enter_frame", "toribashai_position_test", on_enter_frame)
