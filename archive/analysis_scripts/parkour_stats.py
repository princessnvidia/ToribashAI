#!/usr/bin/env python3
from pathlib import Path
import json
import math
from statistics import mean

BASE = Path.home() / "Documents/ToribashAI"
PARKOUR_DIR = BASE / "datasets" / "parkour_json"
OUT = BASE / "metadata" / "parkour_stats.jsonl"
SUMMARY = BASE / "metadata" / "parkour_stats_summary.json"

# Body part 0 semble être une bonne référence centrale au début.
BODY_INDEX = 0
PLAYER_ID = "0"


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def get_pos(frame, player_id=PLAYER_ID, body_index=BODY_INDEX):
    player = frame.get("players", {}).get(player_id)
    if not player:
        return None

    pos = player.get("pos", [])
    if len(pos) <= body_index:
        return None

    return pos[body_index]


def analyze_file(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames", {})
    frame_ids = sorted((int(k) for k in frames.keys()))

    positions = []

    for fid in frame_ids:
        p = get_pos(frames[str(fid)])
        if p is not None:
            positions.append((fid, p))

    if len(positions) < 2:
        return None

    first_fid, first = positions[0]
    last_fid, last = positions[-1]

    step_distances = [
        dist(positions[i - 1][1], positions[i][1])
        for i in range(1, len(positions))
    ]

    ys = [p[1] for _, p in positions]
    zs = [p[2] for _, p in positions]

    total_path = sum(step_distances)
    displacement = dist(first, last)

    return {
        "file": path.name,
        "json_path": str(path),
        "rpl_path": data.get("file"),
        "fightname": data.get("metadata", {}).get("fightname", ""),
        "mod": data.get("metadata", {}).get("mod", ""),
        "frames": len(frame_ids),
        "tracked_positions": len(positions),
        "first_frame": first_fid,
        "last_frame": last_fid,
        "start_pos": first,
        "end_pos": last,
        "displacement_3d": displacement,
        "path_length_3d": total_path,
        "delta_y": last[1] - first[1],
        "delta_z": last[2] - first[2],
        "max_z": max(zs),
        "min_z": min(zs),
        "height_range": max(zs) - min(zs),
        "y_range": max(ys) - min(ys),
        "avg_step_distance": mean(step_distances),
    }


def main():
    files = sorted(PARKOUR_DIR.glob("*.json"))

    rows = []
    errors = 0

    for i, path in enumerate(files, start=1):
        try:
            row = analyze_file(path)
            if row:
                rows.append(row)
        except Exception as e:
            errors += 1
            print("Erreur:", path.name, e)

        if i % 50 == 0:
            print(f"{i}/{len(files)} analysés")

    OUT.parent.mkdir(parents=True, exist_ok=True)

    with OUT.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "files_found": len(files),
        "files_analyzed": len(rows),
        "errors": errors,
        "avg_frames": mean([r["frames"] for r in rows]) if rows else 0,
        "avg_displacement_3d": mean([r["displacement_3d"] for r in rows]) if rows else 0,
        "avg_path_length_3d": mean([r["path_length_3d"] for r in rows]) if rows else 0,
        "avg_height_range": mean([r["height_range"] for r in rows]) if rows else 0,
        "top_distance": sorted(
            rows,
            key=lambda r: r["displacement_3d"],
            reverse=True
        )[:20],
        "top_height_range": sorted(
            rows,
            key=lambda r: r["height_range"],
            reverse=True
        )[:20],
        "longest_replays": sorted(
            rows,
            key=lambda r: r["frames"],
            reverse=True
        )[:20],
    }

    SUMMARY.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nTerminé.")
    print("Stats JSONL:", OUT)
    print("Résumé:", SUMMARY)
    print("Parkours analysés:", len(rows))
    print("Erreurs:", errors)
    print("Frames moyennes:", summary["avg_frames"])
    print("Distance moyenne:", summary["avg_displacement_3d"])
    print("Hauteur moyenne:", summary["avg_height_range"])


if __name__ == "__main__":
    main()
