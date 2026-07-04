echo("[TEST CONTROL V22] loaded")

local filename = "toribashai_control_v22.txt"

local function try_read(root)
local f = io.open(filename, "r", root)
if not f then return false end

    local txt = f:read("*a")
    f:close()

    echo("[TEST CONTROL V22] FOUND root=" .. tostring(root))
    echo("[TEST CONTROL V22] txt=" .. tostring(txt))

    return true
    end

    local function on_draw2d()
    try_read(nil)
    try_read(0)
    try_read(1)
    try_read(2)
    try_read(3)
    end

    remove_hooks("test_control_v22")
    add_hook("draw2d", "test_control_v22", on_draw2d)
