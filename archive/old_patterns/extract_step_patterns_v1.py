#!/usr/bin/env python3
from pathlib import Path
import json
import math
from statistics import mean
from collections import Counter

PROJECT = Path.home() / "Documents" / "ToribashAI"

IN_DIR = PROJECT / "datasets" / "parkour_json"
OUT_DIR = PROJECT / "datasets" / "motion_patterns"

OUT_JSONL = OUT_DIR / "step_patterns_v1.jsonl"
OUT_SUMMARY = OUT_DIR / "step_patterns_v1_summary.json"

PLAYER_ID = "0"
WINDOW_SIZE = 20
STRIDE = 2

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
    elif isinstance(obj, dict):
        for item in obj.values():
            nums.extend(flatten_numbers(item))
    return nums


def points_from_pos(pos):
    nums = flatten_numbers(pos)
    points = []
    for i in range(0, len(nums) - 2, 3):
        points.append({"x": nums[i], "y": nums[i + 1], "z": nums[i + 2], "idx": i // 3})
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
    p = get_player(frame)
    return p.get("pos") or p.get("POS") or p.get("positions") or p.get("body_pos")


def get_actions(frame):
    p = get_player(frame)
    arr = [0] * 20

    joints = p.get("joints", {})
    if isinstance(joints, dict):
        for k, v in joints.items():
            try:
                jid = int(k)
                if 0 <= jid < 20:
                    arr[jid] = int(v)
            except Exception:
                pass

    pairs = p.get("joint_pairs")
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
    vals = [actions[i] for i in ids]
    return sum(1 for v in vals if v not in (0, 3)) / max(1, len(vals))


def load_replay(path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    frames_raw = data.get("frames", {})
    frames = []

    if isinstance(frames_raw, dict):
        for fid, fr in frames_raw.items():
            if isinstance(fr, dict):
                item = dict(fr)
                item["frame"] = int(fid)
                frames.append(item)
    elif isinstance(frames_raw, list):
        frames = [f for f in frames_raw if isinstance(f, dict)]

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

    dy = end["y"] - start["y"]
    dx = end["x"] - start["x"]
    dz = end["z"] - start["z"]

    # Y négatif = direction naturelle de nos meilleurs replays
    forward = -dy

    centers = [w["center"] for w in window]
    z_min = min(c["z"] for c in centers)
    z_max = max(c["z"] for c in centers)
    z_range = z_max - z_min

    leg_act = mean(activity(w["actions"], LEG_JOINTS) for w in window)
    core_act = mean(activity(w["actions"], CORE_JOINTS) for w in window)

    # Approx reverse-engineering des appuis :
    # on prend les 2 points les plus bas du corps à chaque frame.
    low_indices = []
    low_x = []
    low_y = []
    low_z = []

    for w in window:
        lows = lowest_two(w["points"])
        if lows:
            low_indices.append(tuple(sorted(p["idx"] for p in lows)))
            low_x.extend([p["x"] for p in lows])
            low_y.extend([p["y"] for p in lows])
            low_z.extend([p["z"] for p in lows])

    support_changes = 0
    for i in range(1, len(low_indices)):
        if low_indices[i] != low_indices[i - 1]:
            support_changes += 1

    support_change_rate = support_changes / max(1, len(low_indices) - 1)

    # Écart gauche/droite approximé sur X des appuis bas.
    foot_width = (max(low_x) - min(low_x)) if low_x else 0.0

    # Lean forward :
    # on compare les bodyparts les plus hauts aux plus bas.
    lean_values = []
    for w in window:
        pts = sorted(w["points"], key=lambda p: p["z"])
        if len(pts) >= 8:
            lower = center(pts[:6])
            upper = center(pts[-6:])
            # avancer en Y négatif => torse penché vers Y plus négatif que le bas du corps
            lean_values.append(lower["y"] - upper["y"])

    forward_lean = mean(lean_values) if lean_values else 0.0

    step_score = (
        forward * 3.0
        + max(0.0, forward_lean) * 2.0
        + leg_act * 5.0
        + support_change_rate * 3.0
        + min(foot_width, 5.0) * 0.5
        - abs(dx) * 0.8
        - abs(dz) * 0.8
        - max(0.0, z_range - 8.0) * 1.2
    )

    is_step = (
        forward > 1.2
        and leg_act > 0.08
        and forward_lean > 0.05
        and support_change_rate > 0.05
        and z_min > 4.0
        and z_range < 10.0
    )

    return {
        "delta_x": dx,
        "delta_y": dy,
        "delta_z": dz,
        "forward_y_negative": forward,
        "z_min": z_min,
        "z_max": z_max,
        "z_range": z_range,
        "leg_activity": leg_act,
        "core_activity": core_act,
        "support_change_rate": support_change_rate,
        "foot_width": foot_width,
        "forward_lean": forward_lean,
        "step_score": step_score,
        "is_step": is_step,
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

    for i in range(0, len(usable) - WINDOW_SIZE + 1, STRIDE):
        window = usable[i:i + WINDOW_SIZE]
        feat = analyze_window(window)

        if not feat["is_step"]:
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

    for idx, path in enumerate(files, 1):
        try:
            rows = extract_from_file(path)
            all_rows.extend(rows)
            print(f"[{idx}/{len(files)}] {path.name}: {len(rows)} step windows")
        except Exception as e:
            errors.append({"file": str(path), "error": repr(e)})
            print(f"[ERREUR] {path.name}: {e}")

    all_rows.sort(key=lambda r: r["features"]["step_score"], reverse=True)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "files": len(files),
        "step_windows": len(all_rows),
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "errors": errors[:30],
        "error_count": len(errors),
        "top_30": [
            {
                "source_name": r["source_name"],
                "start_frame": r["start_frame"],
                "end_frame": r["end_frame"],
                "delta_y": r["features"]["delta_y"],
                "forward_lean": r["features"]["forward_lean"],
                "support_change_rate": r["features"]["support_change_rate"],
                "leg_activity": r["features"]["leg_activity"],
                "z_min": r["features"]["z_min"],
                "z_range": r["features"]["z_range"],
                "step_score": r["features"]["step_score"],
            }
            for r in all_rows[:30]
        ],
    }

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("Terminé.")
    print(f"Step windows: {len(all_rows)}")
    print(f"JSONL: {OUT_JSONL}")
    print(f"Summary: {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
