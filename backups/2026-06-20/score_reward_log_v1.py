#!/usr/bin/env python3
from pathlib import Path
import re
import json
import math
from statistics import mean

PROJECT = Path.home() / "Documents" / "ToribashAI"

LOG = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script/toribashai_reward_probe_log.txt"

OUT_JSON = PROJECT / "models" / "reward_score_v1.json"

LINE_RE = re.compile(
    r"frame=(?P<frame>-?\d+)\s+"
    r"chest=(?P<cx>-?\d+(?:\.\d+)?),(?P<cy>-?\d+(?:\.\d+)?),(?P<cz>-?\d+(?:\.\d+)?)\s+"
    r"stomach=(?P<sx>-?\d+(?:\.\d+)?),(?P<sy>-?\d+(?:\.\d+)?),(?P<sz>-?\d+(?:\.\d+)?)\s+"
    r"groin=(?P<gx>-?\d+(?:\.\d+)?),(?P<gy>-?\d+(?:\.\d+)?),(?P<gz>-?\d+(?:\.\d+)?)\s+"
    r"lfoot=(?P<lx>-?\d+(?:\.\d+)?),(?P<ly>-?\d+(?:\.\d+)?),(?P<lz>-?\d+(?:\.\d+)?)\s+"
    r"rfoot=(?P<rx>-?\d+(?:\.\d+)?),(?P<ry>-?\d+(?:\.\d+)?),(?P<rz>-?\d+(?:\.\d+)?)"
)


def point(d, prefix):
    return {
        "x": float(d[prefix + "x"]),
        "y": float(d[prefix + "y"]),
        "z": float(d[prefix + "z"]),
    }


def center3(a, b, c):
    return {
        "x": (a["x"] + b["x"] + c["x"]) / 3.0,
        "y": (a["y"] + b["y"] + c["y"]) / 3.0,
        "z": (a["z"] + b["z"] + c["z"]) / 3.0,
    }


def parse_log():
    samples = []

    if not LOG.exists():
        raise FileNotFoundError(f"Log introuvable: {LOG}")

    with LOG.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue

            d = m.groupdict()

            chest = point(d, "c")
            stomach = point(d, "s")
            groin = point(d, "g")
            lfoot = point(d, "l")
            rfoot = point(d, "r")

            samples.append({
                "frame": int(d["frame"]),
                "chest": chest,
                "stomach": stomach,
                "groin": groin,
                "lfoot": lfoot,
                "rfoot": rfoot,
                "center": center3(chest, stomach, groin),
            })

    return samples


def split_last_continuous_segment(samples):
    if not samples:
        return []

    segments = []
    current = [samples[0]]

    for s in samples[1:]:
        prev = current[-1]

        if s["frame"] > prev["frame"] and s["frame"] - prev["frame"] <= 5:
            current.append(s)
        else:
            segments.append(current)
            current = [s]

    segments.append(current)
    segments.sort(key=len, reverse=True)
    return segments[0]


def dist_xy(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def foot_alt_changes(samples):
    changes = 0
    prev = None

    for s in samples:
        # pied le plus bas = appui probable
        side = "L" if s["lfoot"]["z"] < s["rfoot"]["z"] else "R"

        if prev is not None and side != prev:
            changes += 1

        prev = side

    return changes


def main():
    raw = parse_log()
    samples = split_last_continuous_segment(raw)

    if len(samples) < 10:
        raise RuntimeError("Pas assez de samples valides.")

    first = samples[0]
    last = samples[-1]

    start = first["center"]
    end = last["center"]

    dy = end["y"] - start["y"]
    dx = end["x"] - start["x"]
    dz = end["z"] - start["z"]

    # Direction cible actuelle : Y négatif
    forward = -dy

    centers = [s["center"] for s in samples]
    z_values = [c["z"] for c in centers]
    x_values = [c["x"] for c in centers]

    z_min = min(z_values)
    z_max = max(z_values)
    z_mean = mean(z_values)
    z_range = z_max - z_min

    side_drift = max(x_values) - min(x_values)
    foot_changes = foot_alt_changes(samples)
    foot_change_rate = foot_changes / max(1, len(samples) - 1)

    chest_z_end = last["chest"]["z"]
    groin_z_end = last["groin"]["z"]

    fallen = z_min < 3.5 or chest_z_end < 4.0 or groin_z_end < 3.5

    reward = 0.0

    reward += forward * 12.0
    reward += max(0.0, z_min - 4.5) * 5.0
    reward += foot_change_rate * 30.0

    reward -= abs(dx) * 3.0
    reward -= max(0.0, z_range - 2.5) * 8.0
    reward -= max(0.0, side_drift - 3.0) * 5.0

    if fallen:
        reward -= 80.0

    result = {
        "log": str(LOG),
        "raw_samples": len(raw),
        "samples": len(samples),
        "start_frame": first["frame"],
        "end_frame": last["frame"],
        "start_center": start,
        "end_center": end,
        "delta_x": dx,
        "delta_y": dy,
        "delta_z": dz,
        "forward_y_negative": forward,
        "z_min": z_min,
        "z_max": z_max,
        "z_mean": z_mean,
        "z_range": z_range,
        "side_drift": side_drift,
        "foot_changes": foot_changes,
        "foot_change_rate": foot_change_rate,
        "fallen": fallen,
        "reward": reward,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Reward score terminé.")
    print(f"Samples: {result['samples']}")
    print(f"Frames: {result['start_frame']} -> {result['end_frame']}")
    print(f"Forward Y-: {result['forward_y_negative']:.4f}")
    print(f"Delta X: {result['delta_x']:.4f}")
    print(f"Z min: {result['z_min']:.4f}")
    print(f"Z range: {result['z_range']:.4f}")
    print(f"Foot changes: {result['foot_changes']}")
    print(f"Foot change rate: {result['foot_change_rate']:.4f}")
    print(f"Fallen: {result['fallen']}")
    print(f"Reward: {result['reward']:.4f}")
    print(f"JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
