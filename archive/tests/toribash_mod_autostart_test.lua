echo("[ToribashAI] SCRIPT LOADED")

local launched = false

local function on_new_game()
if launched then
    return
    end

    launched = true
    echo("[ToribashAI] NEW GAME DETECTED, OPENING REPLAY 000")

    open_replay("ToribashAI_run_legs_gru_000.rpl", false)
    set_replay_speed(1, true)
    end

    remove_hooks("toribashai_mod_test")
    add_hook("new_game", "toribashai_mod_test", on_new_game)
