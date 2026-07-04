local CONTROL_JOINTS = {4, 5, 6, 7, 14, 15, 16, 17, 18, 19}

local function apply_agent_action()
if not TORIBASHAI_AGENT or not TORIBASHAI_AGENT.actions then
    hold_all()
    return
    end

    local action_index = math.floor(frame / CONFIG.frames_per_action) + 1
    local action = TORIBASHAI_AGENT.actions[action_index]

    if not action then
        hold_all()
        return
        end

        for i, joint_id in ipairs(CONTROL_JOINTS) do
            local value = action[i] or 3
            set_joint_state(0, joint_id, value, true)
            end

            -- Stabilisation minimum du tronc.
            set_joint_state(0, 0, 3, true)
            set_joint_state(0, 1, 3, true)
            set_joint_state(0, 2, 3, true)
            set_joint_state(0, 3, 1, true)
            set_joint_state(0, 12, 3, true)
            set_joint_state(0, 13, 3, true)
            end
