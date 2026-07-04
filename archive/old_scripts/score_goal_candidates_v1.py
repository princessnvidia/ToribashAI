#!/usr/bin/env python3
from pathlib import Path
import math
import re
import json
import argparse

PROJECT = Path.home() / "Documents" / "ToribashAI"

DEFAULT_CANDIDATE_DIRS = [
    PROJECT / "generated" / "goal_candidates_v1",
    PROJECT / "outputs" / "goal_candidates_v1",
    PROJECT / "replays_generated" / "goal_candidates_v1",
    PROJECT / "goal_candidates_v1",
]

# À ajuster si besoin selon la position exacte de ta cible rouge
DEFAULT_TARGET_X = 30.0
DEFAULT_TARGET_Y = 0.0
DEFAULT_TARGET_Z = 0.0


def find_candidate_dir():
    for d in DEFAULT_CANDIDATE_DIRS:
        if d.exists():
            return d
    raise FileNotFoundError("Aucun dossier goal_candidates_v1 trouvé.")


def parse_pos_line(line):
    nums = re.findall(r"-?\d+(?:\.\d+)?", line)
    values = [float(x) for x in nums]

    # POS contient souvent: POS frame body x y z body x y z...
    if len(values) < 4:
        return []

    # On enlève le numéro de frame si présent
    if line.strip().startswith("POS"):
        values = values[1:]

    points = []
    for i in range(0, len(values) - 2, 3):
        x, y, z = values[i], values[i + 1], values[i + 2]
        points.append((x, y, z))

    return points


def read_final_body_center(rpl_path):
    last_points = None
    last_frame = None

    with open(rpl_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("FRAME"):
                nums = re.findall(r"\d+", stripped)
                if nums:
                    last_frame = int(nums[0])

            elif stripped.startswith("POS"):
                points = parse_pos_line(stripped)
                if points:
                    last_points = points

    if not last_points:
        return None

    cx = sum(p[0] for p in last_points) / len(last_points)
    cy = sum(p[1] for p in last_points) / len(last_points)
    cz = sum(p[2] for p in last_points) / len(last_points)

    return {
        "frame": last_frame,
        "center": [cx, cy, cz],
        "body_points": len(last_points),
    }


def distance_3d(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2 +
        (a[2] - b[2]) ** 2
    )


def distance_xy(a, b):
    return math.sqrt(
        (a[0] - b[0]) ** 2 +
        (a[1] - b[1]) ** 2
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None)
    parser.add_argument("--target-x", type=float, default=DEFAULT_TARGET_X)
    parser.add_argument("--target-y", type=float, default=DEFAULT_TARGET_Y)
    parser.add_argument("--target-z", type=float, default=DEFAULT_TARGET_Z)
    args = parser.parse_args()

    candidate_dir = Path(args.dir).expanduser() if args.dir else find_candidate_dir()
    target = [args.target_x, args.target_y, args.target_z]

    rpls = sorted(candidate_dir.glob("*.rpl"))
    if not rpls:
        raise FileNotFoundError(f"Aucun .rpl trouvé dans {candidate_dir}")

    results = []

    for rpl in rpls:
        data = read_final_body_center(rpl)

        if data is None:
            results.append({
                "file": str(rpl),
                "name": rpl.name,
                "error": "Aucune ligne POS trouvée. Impossible de scorer sans positions simulées.",
            })
            continue

        center = data["center"]

        results.append({
            "file": str(rpl),
            "name": rpl.name,
            "frame": data["frame"],
            "center": center,
            "target": target,
            "distance_xy": distance_xy(center, target),
            "distance_3d": distance_3d(center, target),
            "body_points": data["body_points"],
        })

    valid = [r for r in results if "distance_xy" in r]
    valid.sort(key=lambda r: r["distance_xy"])

    out_json = candidate_dir / "goal_candidates_scores_v1.json"
    out_txt = candidate_dir / "goal_candidates_scores_v1.txt"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "candidate_dir": str(candidate_dir),
            "target": target,
            "count": len(results),
            "valid_count": len(valid),
            "best": valid[0] if valid else None,
            "results": valid + [r for r in results if "error" in r],
        }, f, indent=2, ensure_ascii=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(f"Target: {target}\n")
        f.write(f"Candidates: {len(results)}\n")
        f.write(f"Valid: {len(valid)}\n\n")

        if valid:
            f.write("=== BEST ===\n")
            best = valid[0]
            f.write(f"{best['name']}\n")
            f.write(f"distance_xy: {best['distance_xy']:.4f}\n")
            f.write(f"distance_3d: {best['distance_3d']:.4f}\n")
            f.write(f"center: {best['center']}\n\n")

            f.write("=== RANKING ===\n")
            for i, r in enumerate(valid, start=1):
                f.write(
                    f"{i:02d}. {r['name']} | "
                    f"xy={r['distance_xy']:.4f} | "
                    f"3d={r['distance_3d']:.4f} | "
                    f"center={r['center']}\n"
                )
        else:
            f.write("Aucun candidat valide avec POS.\n")

    print("Score terminé.")
    print(f"Dossier: {candidate_dir}")
    print(f"JSON: {out_json}")
    print(f"TXT: {out_txt}")

    if valid:
        best = valid[0]
        print()
        print("MEILLEUR CANDIDAT:")
        print(best["name"])
        print("distance_xy:", round(best["distance_xy"], 4))
        print("distance_3d:", round(best["distance_3d"], 4))
        print("center:", best["center"])
    else:
        print()
        print("Aucun POS trouvé dans les replays.")
        print("Ça veut dire qu'on a généré des actions, mais pas encore simulé les mouvements.")


if __name__ == "__main__":
    main()
