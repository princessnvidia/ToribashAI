echo("[TEST WRITE] START")

local f = io.open("toribashai_test.txt", "w")

if f then
    f:write("hello")
    f:close()
    echo("[TEST WRITE] SUCCESS")
    else
        echo("[TEST WRITE] FAILED")
        end
