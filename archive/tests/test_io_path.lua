echo("[TEST IO PATH] loaded")

local f = io.open("toribashai_where_am_i.txt", "w")
if f then
    f:write("hello from toribash lua\n")
    f:close()
    echo("[TEST IO PATH] write ok")
    else
        echo("[TEST IO PATH] write failed")
        end
