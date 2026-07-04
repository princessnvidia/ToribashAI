echo("################################################")
echo("[ToribashAI Upright V18] LUA LOADED - BALANCE DETECTION STILLNESS")
echo("################################################")


-- V11 note:
-- Large episode counters are deliberately global, not local upvalues.
-- Toribash Lua errors when one function captures more than 60 upvalues.
-- This keeps sample_posture() / finish_run() under the limit.
local CONFIG = {
    max_frames = 500,
    frames_per_action = 20,
    warmup_frames = 20,

    fall_z = 4.5,

    shoulder_high_z = 6.3,
    head_high_z = 6.8,
    hip_high_z = 5.2,

    result_path = "toribashai_episode_result.json"
}

local AGENT_JOINTS = {
    0, 1, 2, 3,
    4, 5, 6, 7,
    8, 9,
    10, 11, 12, 13,
    14, 15, 16, 17, 18, 19
}

-- ToribashAI mapping used in our project:
-- hips / glutes around 4,5,14,15
-- knees around 6,7,16,17
-- ankles / feet around 18,19
local HIP_JOINTS = {
    [4] = true, [5] = true, [14] = true, [15] = true
}

local KNEE_JOINTS = {
    [6] = true, [7] = true, [16] = true, [17] = true
}

-- V13: arms are the first balance tool.
-- We reward useful arm reactions before knee/hip emergency recovery.
local ARM_JOINTS = {
    [8] = true, [9] = true,
    [10] = true, [11] = true,
    [12] = true, [13] = true
}

local LEG_JOINTS = {
    [4] = true, [5] = true, [6] = true, [7] = true,
    [14] = true, [15] = true, [16] = true, [17] = true,
    [18] = true, [19] = true
}

frame = 0
boot_ticks = 0
running = false
started_physics = false
finished = false

start_x = nil
start_y = nil

shoulder_high_frames = 0
head_high_frames = 0
hip_high_frames = 0

shoulder_z_sum = 0
head_z_sum = 0
hip_z_sum = 0
verticality_sum = 0
leg_activity_sum = 0
hip_activity_sum = 0
knee_activity_sum = 0
posture_samples = 0

facing_good_frames = 0
facing_bad_frames = 0
facing_score_sum = 0
facing_turned_frames = 0

step_recovery_sum = 0
step_recovery_frames = 0
bad_step_frames = 0
foot_spread_sum = 0

-- V5: recovery dynamics
previous_verticality = nil
previous_lean_len = nil
recovery_improve_sum = 0
recovery_improve_frames = 0
recovery_worse_frames = 0
recovery_after_lean_frames = 0
lean_frames = 0

-- V7: forward fall anticipation + knee/hip reflex
forward_lean_frames = 0
backward_lean_frames = 0
early_forward_lean_frames = 0
fast_forward_lean_frames = 0

forward_knee_reaction_frames = 0
forward_hip_reaction_frames = 0
early_knee_reaction_frames = 0
early_hip_reaction_frames = 0
fast_knee_reaction_frames = 0
fast_hip_reaction_frames = 0

successful_knee_recovery_frames = 0
successful_hip_recovery_frames = 0
failed_forward_recovery_frames = 0
delayed_knee_penalty_frames = 0

forward_recovery_score_sum = 0
forward_lean_velocity_sum = 0
previous_forward_lean_value = nil

-- V9: anti exploit posture
forward_crouch_frames = 0
hand_ground_frames = 0
hand_ground_sum = 0
best_hand_z = 999
worst_forward_crouch = 0

-- V13: arms / shoulder balancing
arm_activity_sum = 0
arm_reaction_frames = 0
arm_success_frames = 0
arm_failed_frames = 0
arm_recovery_score_sum = 0
shoulder_level_sum = 0
shoulder_unlevel_frames = 0
previous_arm_signature = nil

-- V16: calm start / don't create your own fall
calm_start_frames = 0
panic_start_frames = 0
stable_calm_start_frames = 0
start_action_change_sum = 0
early_verticality_sum = 0
early_motion_penalty_frames = 0

-- V17: smart stillness for the whole run.
stable_state_frames = 0
smart_still_frames = 0
useless_action_frames = 0
perfect_stability_frames = 0
unstable_action_frames = 0
stable_streak = 0
best_stable_streak = 0
smart_still_score_sum = 0

-- V18 clean: detect true balance and reward doing nothing.
balance_detected_frames = 0
still_when_balanced_frames = 0
moved_while_balanced_frames = 0
best_still_balance_streak = 0
current_still_balance_streak = 0
balance_motion_pressure_sum = 0
balance_still_score_sum = 0

best_shoulder_z = 0
previous_leg_signature = nil
previous_hip_signature = nil
previous_knee_signature = nil

local function load_agent()
    TORIBASHAI_AGENT = nil

    local ok, err = pcall(function()
        dofile("toribashai_agent_current.lua")
    end)

    if not ok then
        echo("[Upright V18] AGENT LOAD ERROR")
        echo(tostring(err))
        return false
    end

    echo("################################################")
    echo("[Upright V18] AGENT RELOADED")
    echo("[Upright V18] agent = " .. tostring(TORIBASHAI_AGENT.name))
    echo("################################################")

    return true
end

local function clamp_joint_value(value)
    value = tonumber(value) or 3
    if value < 1 then value = 3 end
    if value > 4 then value = 4 end
    return value
end

local function hold_all()
    for j = 0, 19 do
        set_joint_state(0, j, 3, true)
    end
end

local function body_pos(player, body)
    local info = get_body_info(player, body)
    if not info then return nil end

    if info.pos then
        return info.pos.x or info.pos[1], info.pos.y or info.pos[2], info.pos.z or info.pos[3]
    end

    if info.x and info.y and info.z then
        return info.x, info.y, info.z
    end

    if info[1] and info[2] and info[3] then
        return info[1], info[2], info[3]
    end

    return nil
end

local function avg_body(ids)
    local TMP_SX, TMP_SY, TMP_SZ, c = 0, 0, 0, 0

    for _, id in ipairs(ids) do
        local ok, x, y, z = pcall(body_pos, 0, id)
        if ok and x and y and z then
            TMP_SX = TMP_SX + x
            TMP_SY = TMP_SY + y
            TMP_SZ = TMP_SZ + z
            c = c + 1
        end
    end

    if c == 0 then return nil end
    return TMP_SX / c, TMP_SY / c, TMP_SZ / c
end

local function get_tori_center()
    return avg_body({0, 1, 2, 3, 4, 5})
end

local function get_head_pos(fallback_x, fallback_y, fallback_z)
    local x, y, z = body_pos(0, 0)
    return x or fallback_x or 0, y or fallback_y or 0, z or fallback_z or 0
end

local function get_shoulder_pos(fallback_x, fallback_y, fallback_z)
    local lx, ly, lz = body_pos(0, 11)
    local rx, ry, rz = body_pos(0, 12)

    if lx and rx then
        return (lx + rx) / 2.0, (ly + ry) / 2.0, (lz + rz) / 2.0
    end

    return fallback_x or 0, fallback_y or 0, fallback_z or 0
end

local function get_shoulder_lr()
    local lx, ly, lz = body_pos(0, 11)
    local rx, ry, rz = body_pos(0, 12)
    return lx, ly, lz, rx, ry, rz
end

local function get_hip_pos(fallback_x, fallback_y, fallback_z)
    local x, y, z = avg_body({4, 5, 14, 15})
    return x or fallback_x or 0, y or fallback_y or 0, z or fallback_z or 0
end

local function get_feet_pos()
    local lfx, lfy, lfz = body_pos(0, 18)
    local rfx, rfy, rfz = body_pos(0, 19)
    return lfx, lfy, lfz, rfx, rfy, rfz
end

local function get_current_action()
    if not TORIBASHAI_AGENT or not TORIBASHAI_AGENT.actions then
        return nil
    end

    local action_index = math.floor((frame - CONFIG.warmup_frames) / CONFIG.frames_per_action) + 1
    if action_index < 1 then action_index = 1 end

    return TORIBASHAI_AGENT.actions[action_index]
end

local function signature_for(action, joint_set)
    if not action then return "" end

    local control_joints = TORIBASHAI_AGENT.control_joints or AGENT_JOINTS
    local parts = {}

    for i, joint_id in ipairs(control_joints) do
        if joint_set[joint_id] then
            table.insert(parts, tostring(action[i] or 3))
        end
    end

    return table.concat(parts, ",")
end

local function leg_signature(action)
    return signature_for(action, LEG_JOINTS)
end

local function hip_signature(action)
    return signature_for(action, HIP_JOINTS)
end

local function knee_signature(action)
    return signature_for(action, KNEE_JOINTS)
end

local function arm_signature(action)
    return signature_for(action, ARM_JOINTS)
end

local function get_facing_score()
    local lsx, lsy, lsz, rsx, rsy, rsz = get_shoulder_lr()
    if not lsx or not rsx then return 0, false end

    local shoulder_dx = rsx - lsx
    local shoulder_dy = rsy - lsy

    local abs_x = math.abs(shoulder_dx)
    local abs_y = math.abs(shoulder_dy)

    -- Good: shoulders mostly left-right on X, body faces target axis Y.
    local facing_score = abs_x - abs_y
    local turned = false

    if abs_y > abs_x * 0.85 then
        turned = true
    end

    return facing_score, turned
end

local function normalized_dot(ax, ay, bx, by)
    local al = math.sqrt(ax * ax + ay * ay)
    local bl = math.sqrt(bx * bx + by * by)

    if al < 0.001 or bl < 0.001 then
        return 0
    end

    return (ax * bx + ay * by) / (al * bl)
end

local function get_step_recovery_score()
    local TMP_CX, TMP_CY, TMP_CZ = get_tori_center()
    if not TMP_CX then return 0, 0 end

    local TMP_HX, TMP_HY, TMP_HZ = get_head_pos(TMP_CX, TMP_CY, TMP_CZ)
    local TMP_PX, TMP_PY, TMP_PZ = get_hip_pos(TMP_CX, TMP_CY, TMP_CZ)
    local lfx, lfy, lfz, rfx, rfy, rfz = get_feet_pos()

    if not TMP_HX or not TMP_PX or not lfx or not rfx then
        return 0, 0
    end

    -- Direction where upper body is falling relative to hips.
    local lean_x = TMP_HX - TMP_PX
    local lean_y = TMP_HY - TMP_PY
    local lean_len = math.sqrt(lean_x * lean_x + lean_y * lean_y)

    -- Foot positions relative to hips.
    local lf_x = lfx - TMP_PX
    local lf_y = lfy - TMP_PY
    local rf_x = rfx - TMP_PX
    local rf_y = rfy - TMP_PY

    local foot_dx = lfx - rfx
    local foot_dy = lfy - rfy
    local foot_spread = math.sqrt(foot_dx * foot_dx + foot_dy * foot_dy)

    -- If it is almost vertical, no step recovery is required.
    if lean_len < 0.35 then
        return 0, foot_spread
    end

    local left_score = normalized_dot(lean_x, lean_y, lf_x, lf_y)
    local right_score = normalized_dot(lean_x, lean_y, rf_x, rf_y)
    local best = math.max(left_score, right_score)

    return best, foot_spread
end

local function get_lowest_hand_z(fallback_z)
    -- Broad arm / hand approximation.
    -- Toribash body ids can vary by build, so we sample both forearm/hand-ish ids.
    local ids = {8, 9, 10, 11, 12, 13}
    local lowest = nil

    for _, id in ipairs(ids) do
        local x, y, z = body_pos(0, id)
        if z then
            if lowest == nil or z < lowest then
                lowest = z
            end
        end
    end

    return lowest or fallback_z or 99
end

local function sample_posture()
    TMP_CX, TMP_CY, TMP_CZ = get_tori_center()
    if not TMP_CX then return end

    TMP_HX, TMP_HY, TMP_HZ = get_head_pos(TMP_CX, TMP_CY, TMP_CZ)
    TMP_SX, TMP_SY, TMP_SZ = get_shoulder_pos(TMP_CX, TMP_CY, TMP_CZ)
    TMP_PX, TMP_PY, TMP_PZ = get_hip_pos(TMP_CX, TMP_CY, TMP_CZ)

    local verticality_xy = math.sqrt((TMP_HX - TMP_PX) * (TMP_HX - TMP_PX) + (TMP_HY - TMP_PY) * (TMP_HY - TMP_PY))
    local lean_len = verticality_xy

    posture_samples = posture_samples + 1

    head_z_sum = head_z_sum + TMP_HZ
    shoulder_z_sum = shoulder_z_sum + TMP_SZ
    hip_z_sum = hip_z_sum + TMP_PZ
    verticality_sum = verticality_sum + verticality_xy

    if TMP_SZ > best_shoulder_z then
        best_shoulder_z = TMP_SZ
    end

    if TMP_SZ >= CONFIG.shoulder_high_z then
        shoulder_high_frames = shoulder_high_frames + 1
    end

    if TMP_HZ >= CONFIG.head_high_z then
        head_high_frames = head_high_frames + 1
    end

    if TMP_PZ >= CONFIG.hip_high_z then
        hip_high_frames = hip_high_frames + 1
    end

    -- Facing target / anti 180°.
    local facing_score, turned = get_facing_score()
    facing_score_sum = facing_score_sum + facing_score

    if facing_score > 0.8 then
        facing_good_frames = facing_good_frames + 1
    else
        facing_bad_frames = facing_bad_frames + 1
    end

    if turned then
        facing_turned_frames = facing_turned_frames + 1
    end

    -- Step recovery: foot should be in direction of falling.
    local step_score, foot_spread = get_step_recovery_score()
    step_recovery_sum = step_recovery_sum + step_score
    foot_spread_sum = foot_spread_sum + foot_spread

    if step_score > 0.35 then
        step_recovery_frames = step_recovery_frames + 1
    elseif step_score < -0.25 then
        bad_step_frames = bad_step_frames + 1
    end

    -- V5: reward actual recovery dynamics.
    -- If body was leaning and verticality becomes smaller, it recovered.
    if previous_verticality ~= nil then
        local improvement = previous_verticality - verticality_xy

        if previous_verticality > 0.75 then
            lean_frames = lean_frames + 1

            if improvement > 0.03 then
                recovery_improve_sum = recovery_improve_sum + improvement
                recovery_improve_frames = recovery_improve_frames + 1

                if verticality_xy < previous_verticality * 0.85 then
                    recovery_after_lean_frames = recovery_after_lean_frames + 1
                end
            elseif improvement < -0.03 then
                recovery_worse_frames = recovery_worse_frames + 1
                recovery_improve_sum = recovery_improve_sum + improvement
            end
        end
    end

    -- V7: forward fall anticipation.
    -- Target is toward negative Y in our flat goal mod.
    -- If head is more negative Y than hips, Tori is starting to fall forward.
    local forward_lean = TMP_PY - TMP_HY

    local is_early_forward_lean = forward_lean > 0.25
    local is_forward_lean = forward_lean > 0.45
    local is_backward_lean = forward_lean < -0.60

    local forward_lean_velocity = 0
    if previous_forward_lean_value ~= nil then
        forward_lean_velocity = forward_lean - previous_forward_lean_value
    end

    local is_fast_forward_lean = forward_lean_velocity > 0.045 and forward_lean > 0.12

    forward_lean_velocity_sum = forward_lean_velocity_sum + forward_lean_velocity

    if is_early_forward_lean then
        early_forward_lean_frames = early_forward_lean_frames + 1
    end

    if is_forward_lean then
        forward_lean_frames = forward_lean_frames + 1
    end

    if is_fast_forward_lean then
        fast_forward_lean_frames = fast_forward_lean_frames + 1
    end

    if is_backward_lean then
        backward_lean_frames = backward_lean_frames + 1
    end

    -- V9: voluntary forward crouch / hand-ground exploit detection.
    -- forward_crouch = head forward + low shoulders/hips + not recovering.
    local hand_z = get_lowest_hand_z(TMP_CZ)
    hand_ground_sum = hand_ground_sum + hand_z

    if hand_z < best_hand_z then
        best_hand_z = hand_z
    end

    -- Ground is around z=5.4 in our flat goal mod, so hands under ~5.65 means support / ground scrape.
    if hand_z < 5.65 then
        hand_ground_frames = hand_ground_frames + 1
    end

    local forward_crouch_strength = 0
    if forward_lean > 0.35 and shoulder_high_frames < posture_samples * 0.75 then
        forward_crouch_strength = forward_lean + math.max(0, 6.35 - TMP_SZ) + math.max(0, 5.35 - TMP_PZ)
    end

    if forward_lean > 0.35 and TMP_SZ < 6.35 and TMP_PZ < 5.35 then
        forward_crouch_frames = forward_crouch_frames + 1
    end

    if forward_crouch_strength > worst_forward_crouch then
        worst_forward_crouch = forward_crouch_strength
    end

    previous_verticality = verticality_xy
    previous_lean_len = lean_len

    -- Activity signatures.
    local action = get_current_action()

    local leg_sig = leg_signature(action)
    local hip_sig = hip_signature(action)
    local knee_sig = knee_signature(action)
    local arm_sig = arm_signature(action)

    if previous_leg_signature ~= nil and leg_sig ~= previous_leg_signature then
        leg_activity_sum = leg_activity_sum + 1
    end

    if previous_hip_signature ~= nil and hip_sig ~= previous_hip_signature then
        hip_activity_sum = hip_activity_sum + 1
    end

    if previous_knee_signature ~= nil and knee_sig ~= previous_knee_signature then
        knee_activity_sum = knee_activity_sum + 1
    end

    local arm_changed = previous_arm_signature ~= nil and arm_sig ~= previous_arm_signature

    if arm_changed then
        arm_activity_sum = arm_activity_sum + 1
    end

    -- V13: shoulders should stay level; arms should react before knees.
    local LSX13, LSY13, LSZ13, RSX13, RSY13, RSZ13 = get_shoulder_lr()
    local shoulder_level_delta = 0
    if LSZ13 and RSZ13 then
        shoulder_level_delta = math.abs(LSZ13 - RSZ13)
        shoulder_level_sum = shoulder_level_sum + shoulder_level_delta
        if shoulder_level_delta > 0.35 then
            shoulder_unlevel_frames = shoulder_unlevel_frames + 1
        end
    end

    -- If body starts leaning, arm movement is a good first reflex only if verticality improves.
    if arm_changed and previous_verticality ~= nil and verticality_xy > 0.45 then
        arm_reaction_frames = arm_reaction_frames + 1

        local arm_improvement = previous_verticality - verticality_xy
        if arm_improvement > 0.03 then
            arm_success_frames = arm_success_frames + 1
            arm_recovery_score_sum = arm_recovery_score_sum + arm_improvement
        elseif arm_improvement < -0.03 then
            arm_failed_frames = arm_failed_frames + 1
            arm_recovery_score_sum = arm_recovery_score_sum + arm_improvement
        end
    end

    -- V7: explicit forward fall -> early knee / hip reflex.
    local knee_changed = previous_knee_signature ~= nil and knee_sig ~= previous_knee_signature
    local hip_changed = previous_hip_signature ~= nil and hip_sig ~= previous_hip_signature

    if is_early_forward_lean and knee_changed then
        early_knee_reaction_frames = early_knee_reaction_frames + 1
    end

    if is_early_forward_lean and hip_changed then
        early_hip_reaction_frames = early_hip_reaction_frames + 1
    end

    if is_fast_forward_lean and knee_changed then
        fast_knee_reaction_frames = fast_knee_reaction_frames + 1
    end

    if is_fast_forward_lean and hip_changed then
        fast_hip_reaction_frames = fast_hip_reaction_frames + 1
    end

    if is_forward_lean and knee_changed then
        forward_knee_reaction_frames = forward_knee_reaction_frames + 1
    end

    if is_forward_lean and hip_changed then
        forward_hip_reaction_frames = forward_hip_reaction_frames + 1
    end

    if is_early_forward_lean and not knee_changed and forward_lean_velocity > 0.06 then
        delayed_knee_penalty_frames = delayed_knee_penalty_frames + 1
    end

    if is_forward_lean and previous_verticality ~= nil then
        local improvement_now = previous_verticality - verticality_xy

        if improvement_now > 0.04 then
            forward_recovery_score_sum = forward_recovery_score_sum + improvement_now

            if knee_changed then
                successful_knee_recovery_frames = successful_knee_recovery_frames + 1
            end

            if hip_changed then
                successful_hip_recovery_frames = successful_hip_recovery_frames + 1
            end
        elseif improvement_now < -0.04 then
            forward_recovery_score_sum = forward_recovery_score_sum + improvement_now
            failed_forward_recovery_frames = failed_forward_recovery_frames + 1
        end
    end

    -- V16: first frames should be calm unless real imbalance appears.
    local total_changed_now = 0
    if previous_leg_signature ~= nil and leg_sig ~= previous_leg_signature then
        total_changed_now = total_changed_now + 1
    end
    if previous_hip_signature ~= nil and hip_sig ~= previous_hip_signature then
        total_changed_now = total_changed_now + 1
    end
    if previous_knee_signature ~= nil and knee_sig ~= previous_knee_signature then
        total_changed_now = total_changed_now + 1
    end
    if previous_arm_signature ~= nil and arm_sig ~= previous_arm_signature then
        total_changed_now = total_changed_now + 1
    end

    if frame <= 80 then
        start_action_change_sum = start_action_change_sum + total_changed_now
        early_verticality_sum = early_verticality_sum + verticality_xy

        if total_changed_now == 0 then
            calm_start_frames = calm_start_frames + 1
        else
            panic_start_frames = panic_start_frames + total_changed_now
        end

        if total_changed_now == 0 and TMP_SZ >= CONFIG.shoulder_high_z and TMP_HZ >= CONFIG.head_high_z and verticality_xy < 0.85 then
            stable_calm_start_frames = stable_calm_start_frames + 1
        end

        if total_changed_now > 1 and verticality_xy < 1.10 then
            early_motion_penalty_frames = early_motion_penalty_frames + 1
        end
    end

    -- V17: smart stillness during the whole simulation.
    -- If already stable, doing nothing is good. Moving while stable is bad.
    -- If unstable, actions are allowed because they may be corrective.
    local hand_z_v17 = get_lowest_hand_z(TMP_CZ)
    local stable_state =
        TMP_SZ >= CONFIG.shoulder_high_z
        and TMP_HZ >= CONFIG.head_high_z
        and verticality_xy < 0.85
        and hand_z_v17 > 5.80
        and TMP_PZ >= CONFIG.hip_high_z

    if stable_state then
        stable_state_frames = stable_state_frames + 1
        stable_streak = stable_streak + 1

        if stable_streak > best_stable_streak then
            best_stable_streak = stable_streak
        end

        if total_changed_now == 0 then
            smart_still_frames = smart_still_frames + 1
            perfect_stability_frames = perfect_stability_frames + 1
            smart_still_score_sum = smart_still_score_sum + 1.0
        else
            useless_action_frames = useless_action_frames + total_changed_now
            smart_still_score_sum = smart_still_score_sum - (total_changed_now * 0.75)
        end
    else
        stable_streak = 0

        if total_changed_now > 0 then
            unstable_action_frames = unstable_action_frames + total_changed_now
        end
    end

    -- V18 clean: if Tori is already balanced, the best action is no new action.
    local hand_z_v18 = get_lowest_hand_z(TMP_CZ)

    local balance_detected =
        TMP_SZ >= CONFIG.shoulder_high_z
        and TMP_HZ >= CONFIG.head_high_z
        and TMP_PZ >= CONFIG.hip_high_z
        and verticality_xy < 0.70
        and hand_z_v18 > 5.85

    if balance_detected then
        balance_detected_frames = balance_detected_frames + 1

        if total_changed_now == 0 then
            still_when_balanced_frames = still_when_balanced_frames + 1
            current_still_balance_streak = current_still_balance_streak + 1
            balance_still_score_sum = balance_still_score_sum + 1.0

            if current_still_balance_streak > best_still_balance_streak then
                best_still_balance_streak = current_still_balance_streak
            end
        else
            moved_while_balanced_frames = moved_while_balanced_frames + 1
            balance_motion_pressure_sum = balance_motion_pressure_sum + total_changed_now
            balance_still_score_sum = balance_still_score_sum - (total_changed_now * 1.25)
            current_still_balance_streak = 0
        end
    else
        current_still_balance_streak = 0
    end

    previous_forward_lean_value = forward_lean

    previous_leg_signature = leg_sig
    previous_hip_signature = hip_sig
    previous_knee_signature = knee_sig
    previous_arm_signature = arm_sig
end

local function apply_agent_action()
    if not TORIBASHAI_AGENT or not TORIBASHAI_AGENT.actions then
        hold_all()
        return
    end

    local action = get_current_action()
    if not action then
        hold_all()
        return
    end

    local control_joints = TORIBASHAI_AGENT.control_joints or AGENT_JOINTS

    for i, joint_id in ipairs(control_joints) do
        local value = clamp_joint_value(action[i] or 3)
        set_joint_state(0, joint_id, value, true)
    end
end

local function write_result(score, x, y, z, shoulder_avg, head_avg, hip_avg, verticality_avg, facing_avg, step_avg, foot_spread_avg, recovery_avg, drift, fell, reason)
    local f = io.open(CONFIG.result_path, "w")
    if not f then return end

    f:write(string.format(
        '{"score": %.4f, "x": %.4f, "y": %.4f, "z": %.4f, "shoulder_avg": %.4f, "head_avg": %.4f, "hip_avg": %.4f, "verticality_avg": %.4f, "facing_avg": %.4f, "step_recovery_avg": %.4f, "step_recovery_frames": %d, "bad_step_frames": %d, "foot_spread_avg": %.4f, "recovery_avg": %.4f, "recovery_improve_sum": %.4f, "recovery_improve_frames": %d, "recovery_worse_frames": %d, "recovery_after_lean_frames": %d, "lean_frames": %d, "forward_lean_frames": %d, "backward_lean_frames": %d, "early_forward_lean_frames": %d, "fast_forward_lean_frames": %d, "forward_knee_reaction_frames": %d, "forward_hip_reaction_frames": %d, "early_knee_reaction_frames": %d, "early_hip_reaction_frames": %d, "fast_knee_reaction_frames": %d, "fast_hip_reaction_frames": %d, "successful_knee_recovery_frames": %d, "successful_hip_recovery_frames": %d, "failed_forward_recovery_frames": %d, "delayed_knee_penalty_frames": %d, "forward_recovery_score_sum": %.4f, "forward_lean_velocity_sum": %.4f, "forward_crouch_frames": %d, "hand_ground_frames": %d, "hand_ground_avg": %.4f, "best_hand_z": %.4f, "worst_forward_crouch": %.4f, "hip_activity": %.4f, "knee_activity": %.4f, "arm_activity": %.4f, "arm_reaction_frames": %d, "arm_success_frames": %d, "arm_failed_frames": %d, "arm_recovery_score_sum": %.4f, "shoulder_level_avg": %.4f, "shoulder_unlevel_frames": %d, "calm_start_frames": %d, "panic_start_frames": %d, "stable_calm_start_frames": %d, "start_action_change_sum": %.4f, "early_motion_penalty_frames": %d, "stable_state_frames": %d, "smart_still_frames": %d, "useless_action_frames": %d, "perfect_stability_frames": %d, "unstable_action_frames": %d, "best_stable_streak": %d, "smart_still_score_sum": %.4f, "balance_detected_frames": %d, "still_when_balanced_frames": %d, "moved_while_balanced_frames": %d, "best_still_balance_streak": %d, "balance_motion_pressure_sum": %.4f, "balance_still_score_sum": %.4f, "facing_good_frames": %d, "facing_bad_frames": %d, "facing_turned_frames": %d, "shoulder_high_frames": %d, "head_high_frames": %d, "hip_high_frames": %d, "leg_activity": %.4f, "best_shoulder_z": %.4f, "drift": %.4f, "fell": %s, "frames": %d, "agent": "%s", "reason": "%s"}\n',
        score, x, y, z,
        shoulder_avg, head_avg, hip_avg, verticality_avg,
        facing_avg, step_avg, step_recovery_frames, bad_step_frames, foot_spread_avg,
        recovery_avg, recovery_improve_sum, recovery_improve_frames, recovery_worse_frames, recovery_after_lean_frames, lean_frames,
        forward_lean_frames, backward_lean_frames, early_forward_lean_frames, fast_forward_lean_frames,
        forward_knee_reaction_frames, forward_hip_reaction_frames,
        early_knee_reaction_frames, early_hip_reaction_frames,
        fast_knee_reaction_frames, fast_hip_reaction_frames,
        successful_knee_recovery_frames, successful_hip_recovery_frames, failed_forward_recovery_frames,
        delayed_knee_penalty_frames, forward_recovery_score_sum, forward_lean_velocity_sum,
        forward_crouch_frames, hand_ground_frames, hand_ground_sum / math.max(posture_samples, 1), best_hand_z, worst_forward_crouch,
        hip_activity_sum, knee_activity_sum, arm_activity_sum, arm_reaction_frames, arm_success_frames, arm_failed_frames, arm_recovery_score_sum, shoulder_level_sum / math.max(posture_samples, 1), shoulder_unlevel_frames, calm_start_frames, panic_start_frames, stable_calm_start_frames, start_action_change_sum, early_motion_penalty_frames, stable_state_frames, smart_still_frames, useless_action_frames, perfect_stability_frames, unstable_action_frames, best_stable_streak, smart_still_score_sum, balance_detected_frames, still_when_balanced_frames, moved_while_balanced_frames, best_still_balance_streak, balance_motion_pressure_sum, balance_still_score_sum,
        facing_good_frames, facing_bad_frames, facing_turned_frames,
        shoulder_high_frames, head_high_frames, hip_high_frames,
        leg_activity_sum, best_shoulder_z, drift,
        tostring(fell), frame,
        tostring(TORIBASHAI_AGENT and TORIBASHAI_AGENT.name or "unknown"),
        tostring(reason)
    ))

    f:close()
end

local function finish_run(reason)
    if finished then return end

    finished = true
    running = false

    local x, y, z = get_tori_center()
    if not x then
        freeze_game()
        return
    end

    local dx = x - (start_x or x)
    local dy = y - (start_y or y)
    local drift = math.sqrt(dx * dx + dy * dy)

    local shoulder_avg = 0
    local head_avg = 0
    local hip_avg = 0
    local verticality_avg = 99
    local facing_avg = 0
    local step_avg = 0
    local foot_spread_avg = 0
    local recovery_avg = 0

    if posture_samples > 0 then
        shoulder_avg = shoulder_z_sum / posture_samples
        head_avg = head_z_sum / posture_samples
        hip_avg = hip_z_sum / posture_samples
        verticality_avg = verticality_sum / posture_samples
        facing_avg = facing_score_sum / posture_samples
        step_avg = step_recovery_sum / posture_samples
        foot_spread_avg = foot_spread_sum / posture_samples
        recovery_avg = recovery_improve_sum / posture_samples
    end

    local fell = z < CONFIG.fall_z

    local score = 0

    -- V12: reward principal upright.
    -- Priorité absolue : tête + épaules hautes longtemps.
    score = score + shoulder_high_frames * 40.0
    score = score + head_high_frames * 34.0
    score = score + hip_high_frames * 9.0

    -- V12: hauteur moyenne renforcée.
    score = score + shoulder_avg * 120.0
    score = score + head_avg * 95.0
    score = score + hip_avg * 45.0

    -- Verticalité tête au-dessus du bassin.
    score = score - verticality_avg * 160.0

    -- V15: Facing target disabled. Rotation is allowed if it helps balance.

    -- Step recovery: put a foot where the body is falling.
    score = score + step_recovery_sum * 55.0
    score = score + step_recovery_frames * 12.0
    score = score - bad_step_frames * 20.0

    -- V5: actual balance recovery.
    score = score + recovery_improve_sum * 450.0
    score = score + recovery_improve_frames * 14.0
    score = score + recovery_after_lean_frames * 22.0
    score = score - recovery_worse_frames * 18.0

    -- V5: encourage useful hip/knee strategy, without spam.
    score = score + hip_activity_sum * 10.0
    score = score + knee_activity_sum * 12.0

    -- V8: successful reflex only.
    -- No free points for leaning forward or merely moving knees.
    -- Points happen only when knee / hip reaction improves verticality.
    score = score + successful_knee_recovery_frames * 70.0
    score = score + successful_hip_recovery_frames * 45.0
    score = score + forward_recovery_score_sum * 650.0
    score = score - failed_forward_recovery_frames * 45.0
    score = score - delayed_knee_penalty_frames * 22.0

    -- Strong anti-exploit: leaning forward repeatedly is bad unless it actually recovers.
    if early_forward_lean_frames > 25 then
        local useful_reactions = successful_knee_recovery_frames + successful_hip_recovery_frames
        if useful_reactions < early_forward_lean_frames * 0.08 then
            score = score - 850.0
        end
    end

    if fast_forward_lean_frames > 10 then
        if successful_knee_recovery_frames < fast_forward_lean_frames * 0.15 then
            score = score - 650.0
        end
    end

    if forward_lean_frames > 20 then
        if successful_knee_recovery_frames < forward_lean_frames * 0.12 then
            score = score - 700.0
        end
    end

    -- Extra penalty if it chooses a permanent forward crouch.
    if early_forward_lean_frames > frame * 0.40 and verticality_avg > 1.25 then
        score = score - 900.0
    end

    -- V9: hard anti-exploit.
    -- Upright must not use hands as ground support.
    if hand_ground_frames > 4 then
        score = score - 450.0
    end

    if hand_ground_frames > 12 then
        score = score - 900.0
    end

    -- Upright must not intentionally stay folded forward.
    if forward_crouch_frames > frame * 0.12 then
        score = score - 900.0
    end

    if forward_crouch_frames > frame * 0.25 then
        score = score - 1000.0
    end

    if worst_forward_crouch > 1.8 then
        score = score - 700.0
    end

    if hip_activity_sum > frame * 0.22 then
        score = score - 220.0
    end

    if knee_activity_sum > frame * 0.22 then
        score = score - 220.0
    end

    -- Avoid permanent split.
    if foot_spread_avg > 5.5 then
        score = score - 350.0
    end

    -- Stabilité horizontale.
    score = score - drift * 40.0

    -- Survivre longtemps.
    score = score + frame * 12.0

    -- V14: multiplier pour préserver les champions qui tiennent longtemps debout.
    -- Si tête + épaules restent hautes la majorité du temps, le score est fortement renforcé.
    -- V15: objectif unique = rester debout, même si Tori pivote.
    local upright_ratio = (head_high_frames + shoulder_high_frames) / math.max(frame * 2, 1)
    score = score * (0.5 + upright_ratio)

    -- V16: calm start.
    -- Best initial behavior is to stay still and not create instability.
    score = score + calm_start_frames * 18.0
    score = score + stable_calm_start_frames * 45.0
    score = score - panic_start_frames * 18.0
    score = score - early_motion_penalty_frames * 35.0

    if frame >= 80 and stable_calm_start_frames < 25 then
        score = score - 750.0
    end

    if start_action_change_sum > 35 then
        score = score - 550.0
    end

    -- V17: smart stillness over the entire episode.
    -- Stable + no action = good. Stable + unnecessary action = bad.
    score = score + stable_state_frames * 10.0
    score = score + smart_still_frames * 28.0
    score = score + perfect_stability_frames * 18.0
    score = score + best_stable_streak * 10.0
    score = score + smart_still_score_sum * 35.0
    score = score - useless_action_frames * 26.0

    if best_stable_streak >= 100 then
        score = score + 1200.0
    end

    if best_stable_streak >= 200 then
        score = score + 2400.0
    end

    if stable_state_frames > frame * 0.50 and useless_action_frames > stable_state_frames * 0.18 then
        score = score - 900.0
    end

    -- V18 clean: balance detected => don't move.
    score = score + balance_detected_frames * 14.0
    score = score + still_when_balanced_frames * 55.0
    score = score + best_still_balance_streak * 24.0
    score = score + balance_still_score_sum * 45.0
    score = score - moved_while_balanced_frames * 70.0
    score = score - balance_motion_pressure_sum * 25.0

    if best_still_balance_streak >= 60 then
        score = score + 1800.0
    end

    if best_still_balance_streak >= 120 then
        score = score + 3600.0
    end

    if balance_detected_frames > 50 and still_when_balanced_frames < balance_detected_frames * 0.45 then
        score = score - 1800.0
    end


    -- Micro-corrections jambes.
    score = score + leg_activity_sum * 8.0

    -- Anti-spam jambes.
    if leg_activity_sum > frame * 0.25 then
        score = score - 250.0
    end

    -- Pénalités strictes.
    if fell then score = score - 1000.0 end
    if shoulder_avg < 6.0 then score = score - 500.0 end
    if hip_avg < 5.0 then score = score - 350.0 end
    if verticality_avg > 2.5 then score = score - 350.0 end
    if verticality_avg > 1.8 and early_forward_lean_frames > frame * 0.25 then score = score - 550.0 end
    if shoulder_high_frames < frame * 0.70 then score = score - 650.0 end
    if head_high_frames < frame * 0.65 then score = score - 550.0 end
    -- V15: facing target disabled for pure upright.

    -- V10: hard declass for body parts touching / nearly touching ground.
    if reason == "head_ground" then score = score - 1200.0 end
    if reason == "hips_ground" then score = score - 900.0 end
    if reason == "torso_low" then score = score - 700.0 end
    if reason == "hand_ground" then score = score - 600.0 end
    if reason == "forward_crouch" then score = score - 900.0 end

    echo("================================================")
    echo("[Upright V18] SCORE = " .. tostring(score))
    echo(string.format("[Upright V18] shoulder_high=%d head_high=%d hip_high=%d", shoulder_high_frames, head_high_frames, hip_high_frames))
    echo(string.format("[Upright V18] shoulder_avg=%.2f head_avg=%.2f hip_avg=%.2f", shoulder_avg, head_avg, hip_avg))
    echo(string.format("[Upright V18] verticality=%.2f drift=%.2f leg_activity=%.2f", verticality_avg, drift, leg_activity_sum))
    echo(string.format("[Upright V18] facing_avg=%.2f good=%d bad=%d turned=%d", facing_avg, facing_good_frames, facing_bad_frames, facing_turned_frames))
    echo(string.format("[Upright V18] step_avg=%.2f step_frames=%d bad_steps=%d foot_spread=%.2f", step_avg, step_recovery_frames, bad_step_frames, foot_spread_avg))
    echo(string.format("[Upright V18] recovery_avg=%.4f improve_sum=%.2f improve_frames=%d worse_frames=%d after_lean=%d lean_frames=%d", recovery_avg, recovery_improve_sum, recovery_improve_frames, recovery_worse_frames, recovery_after_lean_frames, lean_frames))
    echo(string.format("[Upright V18] hip_activity=%.2f knee_activity=%.2f", hip_activity_sum, knee_activity_sum))
    echo(string.format("[Upright V18] forward=%d early=%d fast=%d knee=%d hip=%d", forward_lean_frames, early_forward_lean_frames, fast_forward_lean_frames, forward_knee_reaction_frames, forward_hip_reaction_frames))
    echo(string.format("[Upright V18] early_knee=%d fast_knee=%d delayed=%d success_knee=%d failed=%d score=%.2f", early_knee_reaction_frames, fast_knee_reaction_frames, delayed_knee_penalty_frames, successful_knee_recovery_frames, failed_forward_recovery_frames, forward_recovery_score_sum))
    echo(string.format("[Upright V18] useful_reflexes=%d raw_reactions=%d", successful_knee_recovery_frames + successful_hip_recovery_frames, forward_knee_reaction_frames + forward_hip_reaction_frames + early_knee_reaction_frames + early_hip_reaction_frames + fast_knee_reaction_frames + fast_hip_reaction_frames))
    echo(string.format("[Upright V18] forward_crouch=%d hand_ground=%d best_hand_z=%.2f worst_crouch=%.2f", forward_crouch_frames, hand_ground_frames, best_hand_z, worst_forward_crouch))
    echo(string.format("[Upright V18] arms activity=%.2f reaction=%d success=%d failed=%d arm_score=%.2f", arm_activity_sum, arm_reaction_frames, arm_success_frames, arm_failed_frames, arm_recovery_score_sum))
    echo(string.format("[Upright V18] shoulders level_avg=%.2f unlevel=%d", shoulder_level_sum / math.max(posture_samples, 1), shoulder_unlevel_frames))
    echo(string.format("[Upright V18] upright_ratio=%.3f frame_bonus=%d", upright_ratio, frame * 12))
    echo("[Upright V18] facing disabled: rotation allowed for balance")
    echo(string.format("[Upright V18] calm_start=%d stable_calm=%d panic=%d early_motion_penalty=%d", calm_start_frames, stable_calm_start_frames, panic_start_frames, early_motion_penalty_frames))
    echo(string.format("[Upright V18] stable=%d smart_still=%d useless=%d perfect=%d best_streak=%d", stable_state_frames, smart_still_frames, useless_action_frames, perfect_stability_frames, best_stable_streak))
    echo(string.format("[Upright V18] balance_detected=%d still=%d moved=%d best_balance_streak=%d pressure=%.2f", balance_detected_frames, still_when_balanced_frames, moved_while_balanced_frames, best_still_balance_streak, balance_motion_pressure_sum))
    echo("[Upright V18] fell=" .. tostring(fell) .. " reason=" .. tostring(reason))
    echo("================================================")

    write_result(score, x, y, z, shoulder_avg, head_avg, hip_avg, verticality_avg, facing_avg, step_avg, foot_spread_avg, recovery_avg, drift, fell, reason)
    freeze_game()
end

local function start_physics_once()
    if started_physics then return end

    started_physics = true

    echo("[Upright V18] STARTING PHYSICS")

    unfreeze_game()
    toggle_game_pause(false)
    step_game(false, false)
    run_frames(1)
    run_frames(10)
end

local function on_new_game()
    local ok = load_agent()
    if not ok then
        running = false
        freeze_game()
        return
    end

    frame = 0
    boot_ticks = 0
    running = true
    started_physics = false
    finished = false

    start_x = nil
    start_y = nil

    shoulder_high_frames = 0
    head_high_frames = 0
    hip_high_frames = 0

    shoulder_z_sum = 0
    head_z_sum = 0
    hip_z_sum = 0
    verticality_sum = 0
    leg_activity_sum = 0
    hip_activity_sum = 0
    knee_activity_sum = 0
    posture_samples = 0

    facing_good_frames = 0
    facing_bad_frames = 0
    facing_score_sum = 0
    facing_turned_frames = 0

    step_recovery_sum = 0
    step_recovery_frames = 0
    bad_step_frames = 0
    foot_spread_sum = 0

    previous_verticality = nil
    previous_lean_len = nil
    recovery_improve_sum = 0
    recovery_improve_frames = 0
    recovery_worse_frames = 0
    recovery_after_lean_frames = 0
    lean_frames = 0

    forward_lean_frames = 0
    backward_lean_frames = 0
    early_forward_lean_frames = 0
    fast_forward_lean_frames = 0

    forward_knee_reaction_frames = 0
    forward_hip_reaction_frames = 0
    early_knee_reaction_frames = 0
    early_hip_reaction_frames = 0
    fast_knee_reaction_frames = 0
    fast_hip_reaction_frames = 0

    successful_knee_recovery_frames = 0
    successful_hip_recovery_frames = 0
    failed_forward_recovery_frames = 0
    delayed_knee_penalty_frames = 0

    forward_recovery_score_sum = 0
    forward_lean_velocity_sum = 0
    previous_forward_lean_value = nil

    forward_crouch_frames = 0
    hand_ground_frames = 0
    hand_ground_sum = 0
    best_hand_z = 999
    worst_forward_crouch = 0

    arm_activity_sum = 0
    arm_reaction_frames = 0
    arm_success_frames = 0
    arm_failed_frames = 0
    arm_recovery_score_sum = 0
    shoulder_level_sum = 0
    shoulder_unlevel_frames = 0
    previous_arm_signature = nil

    calm_start_frames = 0
    panic_start_frames = 0
    stable_calm_start_frames = 0
    start_action_change_sum = 0
    early_verticality_sum = 0
    early_motion_penalty_frames = 0

    stable_state_frames = 0
    smart_still_frames = 0
    useless_action_frames = 0
    perfect_stability_frames = 0
    unstable_action_frames = 0
    stable_streak = 0
    best_stable_streak = 0
    smart_still_score_sum = 0

    balance_detected_frames = 0
    still_when_balanced_frames = 0
    moved_while_balanced_frames = 0
    best_still_balance_streak = 0
    current_still_balance_streak = 0
    balance_motion_pressure_sum = 0
    balance_still_score_sum = 0

    best_shoulder_z = 0
    previous_leg_signature = nil
    previous_hip_signature = nil
    previous_knee_signature = nil

    hold_all()
    unfreeze_game()
end

local function on_draw2d()
    if not running or started_physics then return end

    boot_ticks = boot_ticks + 1
    if boot_ticks >= 30 then
        start_physics_once()
    end
end


local function is_head_ground()
    local TMP_HX, TMP_HY, TMP_HZ = body_pos(0, 0)
    return TMP_HZ ~= nil and TMP_HZ < 5.65
end

local function is_torso_too_low()
    -- Chest / torso approximation.
    local TMP_CX, TMP_CY, TMP_CZ = body_pos(0, 1)
    if TMP_CZ == nil then
        return false
    end

    return TMP_CZ < 5.75
end

local function is_hips_ground()
    local TMP_HX, TMP_HY, TMP_HZ = get_hip_pos(0, 0, 99)
    return TMP_HZ ~= nil and TMP_HZ < 5.55
end

local function is_upper_body_ground_exploit()
    if is_head_ground() then
        return true, "head_ground"
    end

    if is_hips_ground() then
        return true, "hips_ground"
    end

    if is_torso_too_low() then
        return true, "torso_low"
    end

    return false, ""
end

local function on_enter_frame()
    if not running then return end

    frame = frame + 1

    local x, y, z = get_tori_center()

    if x and not start_x then
        start_x = x
        start_y = y
    end

    sample_posture()

    if frame < CONFIG.warmup_frames then
        hold_all()
    else
        apply_agent_action()
        apply_agent_action()
    end

    run_frames(1)

    if frame >= CONFIG.max_frames then
        finish_run("max_frames")
        return
    end

    if x and z < CONFIG.fall_z then
        finish_run("fell")
        return
    end

    -- V10: immediate elimination if head / hips / torso gets too low.
    if frame > 35 then
        local body_ground, body_reason = is_upper_body_ground_exploit()
        if body_ground then
            finish_run(body_reason)
            return
        end
    end

    -- V9: early eliminate strong exploits.
    if frame > 60 and hand_ground_frames > 20 then
        finish_run("hand_ground")
        return
    end

    if frame > 100 and forward_crouch_frames > frame * 0.35 then
        finish_run("forward_crouch")
        return
    end
end

remove_hooks("toribashai_upright_runner_v18")
remove_hooks("toribashai_upright_runner_v18")
remove_hooks("toribashai_upright_runner_v18")
remove_hooks("toribashai_upright_runner_v18")
remove_hooks("toribashai_upright_runner_v18")
remove_hooks("toribashai_upright_runner_v18")
remove_hooks("toribashai_balance_runner_v1")
remove_hooks("toribashai_balance_runner_v2")
remove_hooks("toribashai_balance_runner_v3")
remove_hooks("toribashai_recovery_runner_v1")

add_hook("new_game", "toribashai_upright_runner_v18", on_new_game)
add_hook("draw2d", "toribashai_upright_runner_v18", on_draw2d)
add_hook("enter_frame", "toribashai_upright_runner_v18", on_enter_frame)
