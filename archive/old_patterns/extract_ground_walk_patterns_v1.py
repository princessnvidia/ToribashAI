#!/usr/bin/env python3
from pathlib import Path
import json
from statistics import mean

PROJECT = Path.home() / "Documents" / "ToribashAI"

IN_DIR = PROJECT / "datasets" / "parkour_json"
OUT_DIR = PROJECT / "datasets" / "motion_patterns"

OUT_JSONL = OUT_DIR / "ground_walk_patterns_v1.jsonl"
OUT_SUMMARY = OUT_DIR / "ground_walk_patterns_v1_summary.json"

PLAYER_ID = "0"

WINDOW_SIZE = 20
STRIDE = 2
MAX_ROWS = 2000

LEG_JOINTS = [14, 15, 16, 17, 18, 19]
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


def points_from_pos(pos):
    nums = flatten_numbers(pos)
    points = []

    for i in range(0, len(nums) - 2, 3):
        points.append({
            "x": nums[i],
            "y": nums[i + 1],
            "z": nums[i + 2],
            "idx": i // 3,
        })

    return points


def center(points):
    if not points:
        return None

    return {
        "x": mean(p["x"] for p in points),
        "y": mean(p["y"] for p in points),
        "z": mean(p["z"] for p in points),
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
    arr = [0] * 20

    joints = player.get("joints", {})
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
                    pass

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


def activity(actions, ids):
    vals = [actions[i] for i in ids if 0 <= i < len(actions)]
    if not vals:
        return 0.0

    return sum(1 for v in vals if int(v) not in (0, 3)) / len(vals)


def load_replay(path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    frames_raw = data.get("frames", {})
    frames = []

    if isinstance(frames_raw, dict):
        for fid, fr in frames_raw.items():
            if not isinstance(fr, dict):
                continue

            item = dict(fr)
            try:
                item["frame"] = int(fid)
            except Exception:
                try:
                    item["frame"] = int(item.get("frame", 0))
                except Exception:
                    item["frame"] = 0

            frames.append(item)

    elif isinstance(frames_raw, list):
        for fr in frames_raw:
            if isinstance(fr, dict):
                frames.append(fr)

    frames.sort(key=lambda f: int(f.get("frame", 0)))
    return data, frames


def lowest_two(points):
    if len(points) < 2:
        return []

    return sorted(points, key=lambda p: p["z"])[:2]


def analyze_window(window):
    first = window[0]
    last = window[-1]

    start = first["center"]
    end = last["center"]

    dx = end["x"] - start["x"]
    dy = end["y"] - start["y"]
    dz = end["z"] - start["z"]

    # Dans nos replays, Y négatif semble être la direction naturelle "forward".
    forward = -dy

    centers = [w["center"] for w in window]
    z_values = [c["z"] for c in centers]

    z_min = min(z_values)
    z_max = max(z_values)
    z_mean = mean(z_values)
    z_range = z_max - z_min

    leg_act = mean(activity(w["actions"], LEG_JOINTS) for w in window)
    core_act = mean(activity(w["actions"], CORE_JOINTS) for w in window)

    low_indices = []
    low_x = []
    low_y = []
    low_z = []

    for w in window:
        lows = lowest_two(w["points"])

        if not lows:
            continue

        low_indices.append(tuple(sorted(p["idx"] for p in lows)))
        low_x.extend([p["x"] for p in lows])
        low_y.extend([p["y"] for p in lows])
        low_z.extend([p["z"] for p in lows])

    support_changes = 0
    for i in range(1, len(low_indices)):
        if low_indices[i] != low_indices[i - 1]:
            support_changes += 1

    support_change_rate = support_changes / max(1, len(low_indices) - 1)

    foot_width = (max(low_x) - min(low_x)) if low_x else 0.0

    lean_values = []

    for w in window:
        pts = sorted(w["points"], key=lambda p: p["z"])

        if len(pts) < 8:
            continue

        lower = center(pts[:6])
        upper = center(pts[-6:])

        if lower is None or upper is None:
            continue

        # Forward Y- : torse plus vers Y- que le bas du corps.
        # lower.y - upper.y > 0 => upper.y est plus négatif => penché vers l'avant.
        lean_values.append(lower["y"] - upper["y"])

    forward_lean = mean(lean_values) if lean_values else 0.0

    # Score spécialisé "marche au sol" :
    # On récompense la progression, le lean, l'alternance d'appuis et l'activité jambes.
    # On pénalise fortement l'altitude et les variations verticales.
    ground_walk_score = (
        forward * 3.0
        + max(0.0, forward_lean) * 3.0
        + leg_act * 8.0
        + core_act * 2.0
        + support_change_rate * 5.0
        + min(foot_width, 4.0) * 0.5
        - abs(dx) * 0.8
        - abs(dz) * 1.0
        - z_min * 1.5
        - z_range * 2.0
    )

    is_ground_walk = (
        forward > 1.0
        and leg_act > 0.10
        and forward_lean > 0.05
        and support_change_rate > 0.10
        and z_min >= 4.0
        and z_min <= 8.0
        and z_range <= 3.0
    )

    return {
        "delta_x": dx,
        "delta_y": dy,
        "delta_z": dz,
        "forward_y_negative": forward,
        "z_min": z_min,
        "z_max": z_max,
        "z_mean": z_mean,
        "z_range": z_range,
        "leg_activity": leg_act,
        "core_activity": core_act,
        "support_change_rate": support_change_rate,
        "foot_width": foot_width,
        "forward_lean": forward_lean,
        "ground_walk_score": ground_walk_score,
        "is_ground_walk": is_ground_walk,
    }


def extract_from_file(path):
    data, frames = load_replay(path)

    usable = []

    for fr in frames:
        pts = points_from_pos(get_pos(fr))
        c = center(pts)

        if not pts or c is None:
            continue

        usable.append({
            "frame": int(fr.get("frame", 0)),
            "points": pts,
            "center": c,
            "actions": get_actions(fr),
        })

    rows = []

    if len(usable) < WINDOW_SIZE:
        return rows

    for i in range(0, len(usable) - WINDOW_SIZE + 1, STRIDE):
        window = usable[i:i + WINDOW_SIZE]
        feat = analyze_window(window)

        if not feat["is_ground_walk"]:
            continue

        # Double sécurité : pas de segments aériens.
        if feat["z_min"] > 8.0:
            continue

        if feat["z_range"] > 3.0:
            continue

        rows.append({
            "source_file": str(path),
            "source_name": path.name,
            "mod": data.get("metadata", {}).get("mod"),
            "start_frame": window[0]["frame"],
            "end_frame": window[-1]["frame"],
            "features": feat,
            "actions": [w["actions"] for w in window],
            "centers": [w["center"] for w in window],
        })

    return rows


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(IN_DIR.glob("*.json"))
    all_rows = []
    errors = []

    for idx, path in enumerate(files, start=1):
        try:
            rows = extract_from_file(path)
            all_rows.extend(rows)
            print(f"[{idx}/{len(files)}] {path.name}: {len(rows)} ground-walk windows")

        except Exception as e:
            errors.append({
                "file": str(path),
                "error": repr(e),
            })
            print(f"[ERREUR] {path.name}: {e}")

    all_rows.sort(key=lambda r: r["features"]["ground_walk_score"], reverse=True)
    all_rows = all_rows[:MAX_ROWS]

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "files": len(files),
        "ground_walk_windows": len(all_rows),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "max_rows": MAX_ROWS,
        "filters": {
            "forward_y_negative_min": 1.0,
            "leg_activity_min": 0.10,
            "forward_lean_min": 0.05,
            "support_change_rate_min": 0.10,
            "z_min_range": [4.0, 8.0],
            "z_range_max": 3.0,
        },
        "errors": errors[:30],
        "error_count": len(errors),
        "top_30": [
            {
                "source_name": r["source_name"],
                "start_frame": r["start_frame"],
                "end_frame": r["end_frame"],
                "delta_y": r["features"]["delta_y"],
                "forward_y_negative": r["features"]["forward_y_negative"],
                "forward_lean": r["features"]["forward_lean"],
                "support_change_rate": r["features"]["support_change_rate"],
                "leg_activity": r["features"]["leg_activity"],
                "core_activity": r["features"]["core_activity"],
                "z_min": r["features"]["z_min"],
                "z_range": r["features"]["z_range"],
                "ground_walk_score": r["features"]["ground_walk_score"],
            }
            for r in all_rows[:30]
        ],
    }

    OUT_SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print("Terminé.")
    print(f"Ground walk windows: {len(all_rows)}")
    print(f"JSONL: {OUT_JSONL}")
    print(f"Summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
