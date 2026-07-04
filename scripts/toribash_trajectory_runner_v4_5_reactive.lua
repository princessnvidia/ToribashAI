echo("TRAJECTORY RUNNER V4.3 PEC + CORE + SHOULDER WEIGHT SHIFT")

local running, started = false, false
local frame, command_i = 0, 1
local agent = nil

local start_x, start_y, start_z = 0,0,0
local prev_x, prev_y, prev_z = 0,0,0

local max_speed_y, total_motion, min_z = 0,0,999
local drift_x_max, max_side_tilt = 0,0
local knee_ground_frames, low_body_frames, forward_fall_frames = 0,0,0
local pec_stability_frames, max_pec_diff = 0,0

local prev_pec_diff = 999
local prev_torso_tilt = 999
local pec_recovery_bonus = 0
local hip_lift_bonus = 0
local hip_drop_penalty = 0
local core_recovery_bonus = 0
local core_drop_penalty = 0

local shoulder_shift_bonus = 0
local shoulder_shift_penalty = 0
local shoulder_last_side = ""
local shoulder_last_switch_frame = -999
local shoulder_switches = 0

local step_count, valid_step_count, same_foot_repeat, hop_penalty = 0,0,0,0
local knee_ankle_fail = 0
local last_foot, last_step_frame = "", -999
local left_was_ground, right_was_ground = false,false

local MIN_STEP_GAP = 34
local MAX_STEP_GAP = 105
local FOOT_GROUND_Z = 1.10

local LEFT_KNEE = 16
local RIGHT_KNEE = 17
local LEFT_ANKLE = 18
local RIGHT_ANKLE = 19
local KNEE_ABOVE_ANKLE_MARGIN = 0.10

local SHOULDER_MIN_DIFF = 0.12
local SHOULDER_MAX_SAFE_DIFF = 0.55
local SHOULDER_MIN_SWITCH_GAP = 12
local SHOULDER_MAX_STUCK_FRAMES = 60

local function load_agent()
    TORIBASHAI_AGENT = nil
    local ok, err = pcall(function() dofile("toribashai_agent_current.lua") end)
    if not ok then
        echo("AGENT LOAD ERROR " .. tostring(err))
        agent = { name="error", commands={}, loop_length=428 }
        return
    end
    agent = TORIBASHAI_AGENT or { name="nil", commands={}, loop_length=428 }
    echo("AGENT " .. tostring(agent.name) .. " commands=" .. tostring(#agent.commands))
end

local function body_pos(player, body)
    local info = get_body_info(player, body)
    if not info then return nil end
    if info.pos then return info.pos.x or info.pos[1], info.pos.y or info.pos[2], info.pos.z or info.pos[3] end
    if info.x and info.y and info.z then return info.x, info.y, info.z end
    if info[1] and info[2] and info[3] then return info[1], info[2], info[3] end
    return nil
end

local function center()
    local sx, sy, sz, c = 0,0,0,0
    for _, id in ipairs({0,1,2,3,4,5}) do
        local ok,x,y,z = pcall(body_pos, 0, id)
        if ok and x and y and z then
            sx, sy, sz, c = sx+x, sy+y, sz+z, c+1
        end
    end
    if c == 0 then return 0,0,0 end
    return sx/c, sy/c, sz/c
end

local function part_z(id)
    local _,_,z = body_pos(0,id)
    return z or 999
end

local function part_y(id)
    local _,y,_ = body_pos(0,id)
    return y or 0
end

local function fall_z()
    local minv = 999
    for _, id in ipairs({0,1,2,3}) do
        local ok,x,y,z = pcall(body_pos,0,id)
        if ok and z and z < minv then minv = z end
    end
    return minv
end

local function torso_z()
    return math.min(part_z(0), part_z(1), part_z(2), part_z(3))
end

local function side_tilt()
    local _,_,z1 = body_pos(0,4)
    local _,_,z2 = body_pos(0,5)
    if not z1 or not z2 then return 0 end
    return math.abs(z1-z2)
end

local function pec_balance()
    local _,_,zl = body_pos(0,4)
    local _,_,zr = body_pos(0,5)
    if not zl or not zr then return 999 end
    return math.abs(zl-zr)
end

local function forward_fall_amount()
    local hip_y = part_y(0)
    local chest_y = part_y(2)
    local head_y = part_y(1)
    local chest_z = part_z(2)
    local head_z = part_z(1)
    local forward = math.max(chest_y - hip_y, head_y - hip_y)
    local low = math.min(chest_z, head_z)
    if forward > 4.0 and low < 6.0 then return forward end
    return 0
end

local function knees_ground()
    return part_z(16) < 0.70 or part_z(17) < 0.70
end

local function knee_above_ankle_for_foot(foot)
    local knee_z = 999
    local ankle_z = 999

    if foot == "L" then
        knee_z = part_z(LEFT_KNEE)
        ankle_z = part_z(LEFT_ANKLE)
    elseif foot == "R" then
        knee_z = part_z(RIGHT_KNEE)
        ankle_z = part_z(RIGHT_ANKLE)
    else
        return false
    end

    return knee_z > ankle_z + KNEE_ABOVE_ANKLE_MARGIN
end

local function register_step(foot)
    local gap = frame - last_step_frame
    step_count = step_count + 1

    if not knee_above_ankle_for_foot(foot) then
        knee_ankle_fail = knee_ankle_fail + 1
        last_foot = foot
        last_step_frame = frame
        return
    end

    if last_foot == "" then
        valid_step_count = valid_step_count + 1
        last_foot = foot
        last_step_frame = frame
        return
    end

    if foot == last_foot then
        same_foot_repeat = same_foot_repeat + 1
        hop_penalty = hop_penalty + 2
        last_step_frame = frame
        return
    end

    if gap < MIN_STEP_GAP then
        hop_penalty = hop_penalty + 3
        last_foot = foot
        last_step_frame = frame
        return
    end

    if gap > MAX_STEP_GAP then
        hop_penalty = hop_penalty + 1
    end

    valid_step_count = valid_step_count + 1
    last_foot = foot
    last_step_frame = frame
end

local function update_steps()
    local left_ground = part_z(18) < FOOT_GROUND_Z
    local right_ground = part_z(19) < FOOT_GROUND_Z

    if left_ground and not left_was_ground then register_step("L") end
    if right_ground and not right_was_ground then register_step("R") end

    left_was_ground = left_ground
    right_was_ground = right_ground
end

local function update_shoulder_weight_shift(pec_diff)
    local lpec_z = part_z(4)
    local rpec_z = part_z(5)
    local side = ""

    if pec_diff < SHOULDER_MIN_DIFF then
        -- Pecs presque égaux : bon équilibre, mais pas assez d'alternance détectable.
        shoulder_shift_bonus = shoulder_shift_bonus + 0.25
        return
    end

    if pec_diff > SHOULDER_MAX_SAFE_DIFF then
        -- Trop d'écart : ce n'est plus une recalibration, c'est une perte d'équilibre.
        shoulder_shift_penalty = shoulder_shift_penalty + 1
        return
    end

    if lpec_z > rpec_z then
        side = "L_HIGH"
    else
        side = "R_HIGH"
    end

    if shoulder_last_side == "" then
        shoulder_last_side = side
        shoulder_last_switch_frame = frame
        return
    end

    if side ~= shoulder_last_side then
        local gap = frame - shoulder_last_switch_frame

        if gap >= SHOULDER_MIN_SWITCH_GAP then
            shoulder_switches = shoulder_switches + 1
            shoulder_shift_bonus = shoulder_shift_bonus + 2.0

            -- Bonus extra si le transfert arrive pendant une phase encore stable.
            if pec_diff < 0.35 and torso_z() > 5.8 then
                shoulder_shift_bonus = shoulder_shift_bonus + 1.5
            end
        else
            shoulder_shift_penalty = shoulder_shift_penalty + 1
        end

        shoulder_last_side = side
        shoulder_last_switch_frame = frame
    else
        if frame - shoulder_last_switch_frame > SHOULDER_MAX_STUCK_FRAMES then
            shoulder_shift_penalty = shoulder_shift_penalty + 1
            shoulder_last_switch_frame = frame
        end
    end
end

local function apply_command(cmd)
    if not cmd or not cmd.pairs then return end
    for _, pair in ipairs(cmd.pairs) do
        local j, v = pair[1], pair[2]
        if j and v and v ~= 0 then
            set_joint_state(0, j, v, true)
        end
    end
end

local function update_metrics()
    local x,y,z = center()
    local vx,vy,vz = x-prev_x, y-prev_y, z-prev_z
    prev_x, prev_y, prev_z = x,y,z

    local progress_x = x - start_x
    local tilt = side_tilt()
    local pec_diff = pec_balance()
    local ffall = forward_fall_amount()

    if vy > max_speed_y then max_speed_y = vy end
    total_motion = total_motion + math.sqrt(vx*vx + vy*vy + vz*vz)
    if z < min_z then min_z = z end
    if math.abs(progress_x) > drift_x_max then drift_x_max = math.abs(progress_x) end
    if tilt > max_side_tilt then max_side_tilt = tilt end

    if frame > 126 then
        if pec_diff > max_pec_diff then max_pec_diff = pec_diff end
        if pec_diff < 0.35 then pec_stability_frames = pec_stability_frames + 1 end
        if torso_z() < 5.2 then low_body_frames = low_body_frames + 1 end
        if ffall > 0 then forward_fall_frames = forward_fall_frames + 1 end

        local hip_z = part_z(0)

        if prev_pec_diff < 999 and pec_diff < prev_pec_diff then
            pec_recovery_bonus = pec_recovery_bonus + ((prev_pec_diff - pec_diff) * 120.0)
        end

        if pec_diff > 0.35 and hip_z > 7.0 then
            hip_lift_bonus = hip_lift_bonus + 1
        end

        if pec_diff > 0.35 and hip_z < 6.0 then
            hip_drop_penalty = hip_drop_penalty + 1
        end

        local torso_tilt_now = tilt
        local torso_height_now = torso_z()

        if prev_torso_tilt < 999 and torso_tilt_now < prev_torso_tilt then
            core_recovery_bonus = core_recovery_bonus + ((prev_torso_tilt - torso_tilt_now) * 100.0)
        end

        if torso_tilt_now > 0.35 and torso_height_now > 5.8 then
            core_recovery_bonus = core_recovery_bonus + 1.5
        end

        if torso_tilt_now > 0.50 and torso_height_now < 5.2 then
            core_drop_penalty = core_drop_penalty + 1
        end

        update_shoulder_weight_shift(pec_diff)

        prev_pec_diff = pec_diff
        prev_torso_tilt = torso_tilt_now
    end

    update_steps()
    if knees_ground() then knee_ground_frames = knee_ground_frames + 1 end
end

local function write_result(reason)
    local x,y,z = center()
    local progress_y = y - start_y
    local progress_x = x - start_x

    local height_penalty = 0
    if min_z < 10 then height_penalty = (10 - min_z) * 18 end

    local backward_penalty = 0
    if progress_y < 0 then backward_penalty = math.abs(progress_y) * 60 end

    local score =
        valid_step_count * 70.0 +
        pec_stability_frames * 55.0 +
        pec_recovery_bonus * 1.0 +
        core_recovery_bonus * 1.0 +
        shoulder_shift_bonus * 35.0 -
        shoulder_shift_penalty * 90.0 +
        hip_lift_bonus * 30.0 -
        hip_drop_penalty * 80.0 -
        core_drop_penalty * 90.0 +
        progress_y * 0.5 +
        max_speed_y * 2.0 -
        same_foot_repeat * 140.0 -
        hop_penalty * 120.0 -
        knee_ankle_fail * 140.0 -
        math.abs(progress_x) * 12.0 -
        drift_x_max * 10.0 -
        max_side_tilt * 25.0 -
        max_pec_diff * 200.0 -
        knee_ground_frames * 60.0 -
        low_body_frames * 50.0 -
        forward_fall_frames * 90.0 -
        total_motion * 0.02 -
        height_penalty -
        backward_penalty

    local f = io.open("toribashai_episode_result.json","w")
    if f then
        f:write(string.format(
            '{"score": %.6f, "reason": "%s", "frames": %d, "progress_y": %.6f, "progress_x": %.6f, "max_speed_y": %.6f, "motion": %.6f, "min_z": %.6f, "commands": %d, "steps": %d, "valid_steps": %d, "same_foot": %d, "hop_penalty": %d, "knee_ankle_fail": %d, "bad_knee_ankle_steps": %d, "knee_not_above_ankle": %d, "side_tilt": %.6f, "max_pec_diff": %.6f, "knee_ground": %d, "low_body": %d, "forward_fall": %d, "pec_stability": %d, "pec_recovery_bonus": %.6f, "hip_lift_bonus": %d, "hip_drop_penalty": %d, "core_recovery_bonus": %.6f, "core_drop_penalty": %d, "shoulder_shift_bonus": %.6f, "shoulder_shift_penalty": %d, "shoulder_switches": %d}',
            score, reason, frame, progress_y, progress_x, max_speed_y, total_motion, min_z,
            agent and #agent.commands or 0, step_count, valid_step_count, same_foot_repeat,
            hop_penalty, knee_ankle_fail, knee_ankle_fail, knee_ankle_fail,
            max_side_tilt, max_pec_diff, knee_ground_frames, low_body_frames,
            forward_fall_frames, pec_stability_frames,
            pec_recovery_bonus, hip_lift_bonus, hip_drop_penalty,
            core_recovery_bonus, core_drop_penalty,
            shoulder_shift_bonus, shoulder_shift_penalty, shoulder_switches
        ))
        f:close()
    end

    echo("RESULT " .. tostring(reason) .. " score=" .. tostring(score))
    running = false
    freeze_game()
end

local function reset_runner()
    load_agent()
    running, started = true, false
    frame, command_i = 0, 1

    max_speed_y, total_motion, min_z = 0,0,999
    drift_x_max, max_side_tilt = 0,0
    knee_ground_frames, low_body_frames, forward_fall_frames = 0,0,0
    pec_stability_frames, max_pec_diff = 0,0

    prev_pec_diff = 999
    prev_torso_tilt = 999
    pec_recovery_bonus = 0
    hip_lift_bonus = 0
    hip_drop_penalty = 0
    core_recovery_bonus = 0
    core_drop_penalty = 0

    shoulder_shift_bonus = 0
    shoulder_shift_penalty = 0
    shoulder_last_side = ""
    shoulder_last_switch_frame = -999
    shoulder_switches = 0

    step_count, valid_step_count, same_foot_repeat, hop_penalty = 0,0,0,0
    knee_ankle_fail = 0
    last_foot, last_step_frame = "", -999
    left_was_ground, right_was_ground = false,false

    start_x,start_y,start_z = center()
    prev_x,prev_y,prev_z = start_x,start_y,start_z

    unfreeze_game()
    toggle_game_pause(false)
    echo("V4.3 pec + core + shoulder weight shift reset")
end

add_hook("new_game", "trajectory_v43_weightshift_newgame", reset_runner)

add_hook("draw2d", "trajectory_v43_weightshift_draw2d", function()
    if not running then return end
    unfreeze_game()
    toggle_game_pause(false)

    if not started then
        started = true
        step_game(false, false)
        run_frames(10)
        return
    end

    run_frames(1)
end)

add_hook("enter_frame", "trajectory_v43_weightshift_frame", function()
    if not running then return end
    frame = frame + 1

    while command_i <= #agent.commands and agent.commands[command_i].frame <= frame do
        apply_command(agent.commands[command_i])
        command_i = command_i + 1
    end

    update_metrics()

    if frame > 90 and fall_z() < 0.40 then
        write_result("fallen")
        return
    end

    if frame >= 428 then
        write_result("xioi_commands_done")
        return
    end
end)

echo("TRAJECTORY RUNNER V4.3 PEC + CORE + SHOULDER WEIGHT SHIFT READY")
