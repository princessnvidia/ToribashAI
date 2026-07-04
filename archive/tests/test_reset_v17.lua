echo("[TEST RESET V17] loaded")

local ticks = 0

local function on_draw2d()
ticks = ticks + 1

if ticks == 120 then
    echo("[TEST RESET V17] trying runCmd reset true")
    local ok, err = pcall(function()
    runCmd("reset", true)
    end)
    echo("[TEST RESET V17] runCmd reset true ok = " .. tostring(ok))
    if not ok then echo(tostring(err)) end
        end
        end

        remove_hooks("test_reset_v17")
        add_hook("draw2d", "test_reset_v17", on_draw2d)
