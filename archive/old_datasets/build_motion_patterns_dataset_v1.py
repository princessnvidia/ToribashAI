#!/usr/bin/env python3
from pathlib import Path
import json
import math
from statistics import mean

PROJECT = Path.home() / "Documents" / "ToribashAI"

IN_DIR = PROJECT / "datasets" / "parkour_json"
OUT_DIR = PROJECT / "datasets" / "motion_patterns"

OUT_JSONL = OUT_DIR / "motion_patterns_v1.jsonl"
OUT_SUMMARY = OUT_DIR / "motion_patterns_v1_summary.json"

WINDOW_SIZE = 16
STRIDE = 4
PLAYER_ID = "0"

LEG_JOINTS = [14, 15, 16, 17, 18, 19]
ARM_JOINTS = [4, 5, 6, 7, 8, 9, 10, 11]
CORE_JOINTS = [0, 1, 2, 3, 12, 13]


def flatten_numbers(obj):
    nums = []

    if obj is None:
        return nums

    if isinstance(obj, (int, float)):
        return [float(obj)]

    if isinstance(obj, str):
        try:
            return [float(obj)]
        except Exception:
            return []

    if isinstance(obj, list):
        for item in obj:
            nums.extend(flatten_numbers(item))
        return nums

    if isinstance(obj, dict):
        for item in obj.values():
            nums.extend(flatten_numbers(item))
        return nums

    return nums


def center_from_pos(pos):
    nums = flatten_numbers(pos)

    if len(nums) < 3:
        return None

    points = []
    for i in range(0, len(nums) - 2, 3):
        points.append((nums[i], nums[i + 1], nums[i + 2]))

    if not points:
        return None

    return {
        "x": mean(p[0] for p in points),
        "y": mean(p[1] for p in points),
        "z": mean(p[2] for p in points),
    }


def get_player(frame):
    return frame.get("players", {}).get(PLAYER_ID, {})


def get_pos(frame):
    player = get_player(frame)

    return (
        player.get("pos")
        or player.get("POS")
        or player.get("positions")
        or player.get("body_pos")
    )


def get_actions(frame):
    player = get_player(frame)
    joints = player.get("joints", {})

    arr = [0] * 20

    if isinstance(joints, dict):
        for k, v in joints.items():
            try:
                jid = int(k)
                if 0 <= jid < 20:
                    arr[jid] = int(v)
            except Exception:
                pass

    elif isinstance(joints, list):
        if len(joints) == 20:
            for i, v in enumerate(joints):
                try:
                    arr[i] = int(v)
                except Exception:
                    arr[i] = 0

    pairs = player.get("joint_pairs")
    if isinstance(pairs, list):
        for pair in pairs:
            if isinstance(pair, list) and len(pair) >= 2:
                try:
                    jid = int(pair[0])
                    val = int(pair[1])
                    if 0 <= jid < 20:
                        arr[jid] = val
                except Exception:
                    pass

    return arr


def action_activity(actions, joint_ids=None):
    if joint_ids is None:
        joint_ids = range(len(actions))

    vals = [actions[j] for j in joint_ids if 0 <= j < len(actions)]
    if not vals:
        return 0.0

    return sum(1 for v in vals if int(v) not in (0, 3)) / len(vals)


def action_change_rate(seq):
    if len(seq) < 2:
        return 0.0

    changes = 0
    total = 0

    for i in range(1, len(seq)):
        a = seq[i - 1]
        b = seq[i]
        for x, y in zip(a, b):
            total += 1
            if x != y:
                changes += 1

    return changes / max(1, total)


def dist_xy(a, b):
    return math.sqrt((b["x"] - a["x"]) ** 2 + (b["y"] - a["y"]) ** 2)


def load_replay(path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    frames_raw = data.get("frames", {})
    frames = []

    if isinstance(frames_raw, dict):
        for frame_id, frame_data in frames_raw.items():
            if isinstance(frame_data, dict):
                item = dict(frame_data)
                try:
                    item["frame"] = int(frame_id)
                except Exception:
                    item["frame"] = int(item.get("frame", 0))
                frames.append(item)

    elif isinstance(frames_raw, list):
        for item in frames_raw:
            if isinstance(item, dict):
                frames.append(item)

    frames.sort(key=lambda x: int(x.get("frame", 0)))
    return data, frames


def rough_label(features):
    dy = features["delta_y"]
    dz = features["delta_z"]
    speed = features["speed_xy"]
    z_min = features["z_min"]
    z_range = features["z_range"]
    leg = features["leg_activity"]
    arm = features["arm_activity"]
    activity = features["activity"]

    if z_min < 2.0:
        return "fall_or_ground"

    if speed < 0.015 and activity < 0.10:
        return "idle_hold"

    if abs(dz) > 5.0 or z_range > 8.0:
        return "jump_or_trick"

    if dy < -2.0 and leg > 0.10:
        return "forward_y_negative"

    if dy > 2.0 and leg > 0.10:
        return "forward_y_positive"

    if speed > 0.06 and (arm > 0.15 or leg > 0.15):
        return "dynamic_motion"

    return "other_motion"


def extract_windows(path):
    data, frames = load_replay(path)

    usable = []

    for frame in frames:
        center = center_from_pos(get_pos(frame))
        actions = get_actions(frame)

        if center is None:
            continue

        usable.append({
            "frame": int(frame.get("frame", 0)),
            "center": center,
            "actions": actions,
        })

    rows = []

    if len(usable) < WINDOW_SIZE:
        return rows

    for start in range(0, len(usable) - WINDOW_SIZE + 1, STRIDE):
        window = usable[start:start + WINDOW_SIZE]

        first = window[0]
        last = window[-1]

        dt = max(1, last["frame"] - first["frame"])

        centers = [w["center"] for w in window]
        actions_seq = [w["actions"] for w in window]

        dx = last["center"]["x"] - first["center"]["x"]
        dy = last["center"]["y"] - first["center"]["y"]
        dz = last["center"]["z"] - first["center"]["z"]

        displacement_xy = dist_xy(first["center"], last["center"])
        speed_xy = displacement_xy / dt

        zs = [c["z"] for c in centers]

        activity_values = [action_activity(a) for a in actions_seq]
        leg_values = [action_activity(a, LEG_JOINTS) for a in actions_seq]
        arm_values = [action_activity(a, ARM_JOINTS) for a in actions_seq]
        core_values = [action_activity(a, CORE_JOINTS) for a in actions_seq]

        features = {
            "delta_x": dx,
            "delta_y": dy,
            "delta_z": dz,
            "displacement_xy": displacement_xy,
            "speed_xy": speed_xy,
            "z_min": min(zs),
            "z_max": max(zs),
            "z_mean": mean(zs),
            "z_range": max(zs) - min(zs),
            "activity": mean(activity_values),
            "leg_activity": mean(leg_values),
            "arm_activity": mean(arm_values),
            "core_activity": mean(core_values),
            "action_change_rate": action_change_rate(actions_seq),
        }

        rows.append({
            "source_file": str(path),
            "source_name": path.name,
            "mod": data.get("metadata", {}).get("mod"),
            "start_frame": first["frame"],
            "end_frame": last["frame"],
            "frames": [w["frame"] for w in window],
            "start_center": first["center"],
            "end_center": last["center"],
            "features": features,
            "rough_label": rough_label(features),
            "actions": actions_seq,
        })

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(IN_DIR.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"Aucun JSON trouvé dans {IN_DIR}")

    total_rows = 0
    label_counts = {}
    errors = []

    with OUT_JSONL.open("w", encoding="utf-8") as out:
        for i, path in enumerate(files, start=1):
            try:
                rows = extract_windows(path)

                for row in rows:
                    out.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_rows += 1
                    label = row["rough_label"]
                    label_counts[label] = label_counts.get(label, 0) + 1

                print(f"[{i}/{len(files)}] {path.name}: {len(rows)} windows")

            except Exception as e:
                errors.append({"file": str(path), "error": repr(e)})
                print(f"[ERREUR] {path.name}: {e}")

    summary = {
        "input_dir": str(IN_DIR),
        "output_jsonl": str(OUT_JSONL),
        "files": len(files),
        "windows": total_rows,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "label_counts": label_counts,
        "errors": errors[:30],
        "error_count": len(errors),
    }

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("Terminé.")
    print(f"Windows: {total_rows}")
    print(f"Labels: {label_counts}")
    print(f"JSONL: {OUT_JSONL}")
    print(f"Summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
