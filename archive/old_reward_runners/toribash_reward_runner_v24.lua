echo("################################################")
echo("[ToribashAI V24] LUA LOADED - MINIMAL CONTROL RESET")
echo("################################################")

local path = "toribashai_control_v24.txt"

local function check_control(origin)
local f = io.open(path, "r")
if not f then
    f = io.open(path, "r", 0)
    end

    if not f then return end

        local cmd = f:read("*a")
        f:close()

        echo("[ToribashAI V24] FILE DETECTED from " .. tostring(origin))
        echo("[ToribashAI V24] cmd = " .. tostring(cmd))

        local ok_remove, err_remove = pcall(function()
        os.remove(path)
        end)

        echo("[ToribashAI V24] remove ok = " .. tostring(ok_remove))
        if not ok_remove then echo("[ToribashAI V24] remove err = " .. tostring(err_remove)) end

            echo("[ToribashAI V24] RUN runCmd reset true")

            local ok, err = pcall(function()
            runCmd("reset", true)
            end)

            echo("[ToribashAI V24] reset ok = " .. tostring(ok))
            if not ok then echo("[ToribashAI V24] reset err = " .. tostring(err)) end
                end

                local function on_draw2d()
                check_control("draw2d")
                end

                local function on_enter_frame()
                check_control("enter_frame")
                end

                local function on_new_game()
                echo("[ToribashAI V24] new_game")
                check_control("new_game")
                end

                remove_hooks("toribashai_reward_runner_v23")
                remove_hooks("toribashai_reward_runner_v24")

                add_hook("draw2d", "toribashai_reward_runner_v24", on_draw2d)
                add_hook("enter_frame", "toribashai_reward_runner_v24", on_enter_frame)
                add_hook("new_game", "toribashai_reward_runner_v24", on_new_game)
