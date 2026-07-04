echo("[TEST CONTROL V21] loaded")

local path = "toribashai_control_v21.txt"

local function check_control(origin)
local f = io.open(path, "r", 1)
if not f then return end

    local cmd = f:read("*a")
    f:close()

    echo("[TEST CONTROL V21] FILE DETECTED from " .. tostring(origin))
    echo("[TEST CONTROL V21] cmd = " .. tostring(cmd))
    echo("[TEST CONTROL V21] RESET TRUE NOW")

    local ok_remove, err_remove = pcall(function()
    os.remove(path)
    end)

    echo("[TEST CONTROL V21] remove ok = " .. tostring(ok_remove))
    if not ok_remove then
        echo("[TEST CONTROL V21] remove err = " .. tostring(err_remove))
        end

        local ok, err = pcall(function()
        runCmd("reset", true)
        end)

        echo("[TEST CONTROL V21] reset ok = " .. tostring(ok))
        if not ok then
            echo("[TEST CONTROL V21] reset err = " .. tostring(err))
            end
            end

            local function on_draw2d()
            check_control("draw2d")
            end

            local function on_enter_frame()
            check_control("enter_frame")
            end

            local function on_new_game()
            echo("[TEST CONTROL V21] new_game")
            check_control("new_game")
            end

            remove_hooks("test_control_v18")
            remove_hooks("test_control_v18b")
            remove_hooks("test_control_v20")
            remove_hooks("test_control_v21")

            add_hook("draw2d", "test_control_v21", on_draw2d)
            add_hook("enter_frame", "test_control_v21", on_enter_frame)
            add_hook("new_game", "test_control_v21", on_new_game)
