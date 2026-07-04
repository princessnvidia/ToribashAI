echo("[TEST CONTROL V18] loaded")

local path = "toribashai_control_v17.txt"
local ticks = 0

local function on_draw2d()
ticks = ticks + 1

if ticks % 60 == 0 then
    echo("[TEST CONTROL V18] alive tick=" .. tostring(ticks))
    end

    local f = io.open(path, "r")
    if not f then return end

        local cmd = f:read("*a")
        f:close()

        echo("[TEST CONTROL V18] FILE DETECTED: " .. tostring(cmd))

        os.remove(path)

        echo("[TEST CONTROL V18] TRY RESET TRUE")

        local ok, err = pcall(function()
        runCmd("reset", true)
        end)

        echo("[TEST CONTROL V18] reset ok=" .. tostring(ok))
        if not ok then echo(tostring(err)) end
            end

            remove_hooks("test_control_v18")
            add_hook("draw2d", "test_control_v18", on_draw2d)
