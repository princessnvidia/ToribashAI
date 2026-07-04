#!/usr/bin/env python3
from pathlib import Path
import re
import math
import json

PROJECT = Path.home() / "Documents" / "ToribashAI"

LOG_PATH = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script/toribashai_probe_log.txt"

OUT_JSON = PROJECT / "models" / "probe_score_v1.json"

TARGET = {
    "x": 0.0,
    "y": -12.0,
    "z": 5.4,
}

LINE_RE = re.compile(
    r"frame=(?P<frame>-?\d+)\s+"
    r"chest=(?P<cx>-?\d+(?:\.\d+)?),(?P<cy>-?\d+(?:\.\d+)?),(?P<cz>-?\d+(?:\.\d+)?)\s+"
    r"stomach=(?P<sx>-?\d+(?:\.\d+)?),(?P<sy>-?\d+(?:\.\d+)?),(?P<sz>-?\d+(?:\.\d+)?)\s+"
    r"groin=(?P<gx>-?\d+(?:\.\d+)?),(?P<gy>-?\d+(?:\.\d+)?),(?P<gz>-?\d+(?:\.\d+)?)"
)


def dist_xy(a, b):
    return math.sqrt((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2)


def dist_3d(a, b):
    return math.sqrt(
        (a["x"] - b["x"]) ** 2 +
        (a["y"] - b["y"]) ** 2 +
        (a["z"] - b["z"]) ** 2
    )


def parse_log(path):
    samples = []

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue

            d = m.groupdict()

            chest = {
                "x": float(d["cx"]),
                "y": float(d["cy"]),
                "z": float(d["cz"]),
            }

            stomach = {
                "x": float(d["sx"]),
                "y": float(d["sy"]),
                "z": float(d["sz"]),
            }

            groin = {
                "x": float(d["gx"]),
                "y": float(d["gy"]),
                "z": float(d["gz"]),
            }

            center = {
                "x": (chest["x"] + stomach["x"] + groin["x"]) / 3.0,
                "y": (chest["y"] + stomach["y"] + groin["y"]) / 3.0,
                "z": (chest["z"] + stomach["z"] + groin["z"]) / 3.0,
            }

            samples.append({
                "frame": int(d["frame"]),
                "chest": chest,
                "stomach": stomach,
                "groin": groin,
                "center": center,
            })

    return samples


def main():
    if not LOG_PATH.exists():
        raise FileNotFoundError(f"Log introuvable: {LOG_PATH}")

    samples = parse_log(LOG_PATH)

    if not samples:
        raise RuntimeError("Aucune position lisible dans le log.")

    first = samples[0]
    last = samples[-1]

    start = first["center"]
    end = last["center"]

    score = {
        "log_path": str(LOG_PATH),
        "sample_count": len(samples),
        "start_frame": first["frame"],
        "end_frame": last["frame"],
        "start_center": start,
        "end_center": end,
        "target": TARGET,
        "distance_xy_to_target": dist_xy(end, TARGET),
        "distance_3d_to_target": dist_3d(end, TARGET),
        "delta_x": end["x"] - start["x"],
        "delta_y": end["y"] - start["y"],
        "delta_z": end["z"] - start["z"],
        "min_z": min(s["center"]["z"] for s in samples),
        "max_z": max(s["center"]["z"] for s in samples),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(score, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Score probe terminé.")
    print(f"Samples: {score['sample_count']}")
    print(f"Frames: {score['start_frame']} -> {score['end_frame']}")
    print()
    print("Départ:", score["start_center"])
    print("Fin:", score["end_center"])
    print("Cible:", TARGET)
    print()
    print("Distance XY cible:", round(score["distance_xy_to_target"], 4))
    print("Distance 3D cible:", round(score["distance_3d_to_target"], 4))
    print("Delta X:", round(score["delta_x"], 4))
    print("Delta Y:", round(score["delta_y"], 4))
    print("Delta Z:", round(score["delta_z"], 4))
    print("Min Z:", round(score["min_z"], 4))
    print()
    print(f"JSON: {OUT_JSON}")


if __name__ == "__main__":
    main()
