echo("[TEST CONTROL V20] loaded")

local path = "toribashai_control_v19.txt"

local function check_control(origin)
local f = io.open(path, "r")
if not f then return end

    local cmd = f:read("*a")
    f:close()
    os.remove(path)

    echo("[TEST CONTROL V20] FILE DETECTED from " .. tostring(origin))
    echo("[TEST CONTROL V20] cmd = " .. tostring(cmd))
    echo("[TEST CONTROL V20] runCmd reset true")

    local ok, err = pcall(function()
    runCmd("reset", true)
    end)

    echo("[TEST CONTROL V20] reset ok = " .. tostring(ok))
    if not ok then echo(tostring(err)) end
        end

        local function on_draw2d()
        check_control("draw2d")
        end

        local function on_enter_frame()
        check_control("enter_frame")
        end

        local function on_new_game()
        echo("[TEST CONTROL V20] new_game")
        check_control("new_game")
        end

        remove_hooks("test_control_v20")

        add_hook("draw2d", "test_control_v20", on_draw2d)
        add_hook("enter_frame", "test_control_v20", on_enter_frame)
        add_hook("new_game", "test_control_v20", on_new_game)
