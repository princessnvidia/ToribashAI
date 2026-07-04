echo("[ToribashAI] AGENT TEST LOADED")

local frame = 0
local running = false

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

            local function on_new_game()
            echo("[ToribashAI] NEW GAME -> START AGENT")

            frame = 0
            running = true

            hold_all()
            unfreeze_game()
            end

            local function on_enter_frame()
            if not running then
                return
                end

                frame = frame + 1

                if frame < 20 then
                    hold_all()
                    else
                        apply_step()
                        end

                        step_game(false, true)

                        if frame >= 600 then
                            running = false
                            freeze_game()
                            echo("[ToribashAI] AGENT TEST DONE")
                            end
                            end

                            remove_hooks("toribashai_agent_test")
                            add_hook("new_game", "toribashai_agent_test", on_new_game)
                            add_hook("enter_frame", "toribashai_agent_test", on_enter_frame)
