#!/usr/bin/env python3
import json
import math
from pathlib import Path
from collections import Counter

ROOT = Path.home() / "Documents" / "ToribashAI"

IN_PATH = ROOT / "datasets/ml/parkour_sequences_len8.jsonl"
OUT_PATH = ROOT / "datasets/ml/walk_motion_v2.jsonl"
SUMMARY_PATH = ROOT / "datasets/ml/walk_motion_v2_summary.json"

PLAYER_ID = "0"

# Action full 20 joints.
LEG_IDXS = [4, 5, 6, 7, 14, 15, 16, 17, 18, 19]

# On utilise le centre moyen du corps, plus robuste qu'un seul bodypart.
CENTER_PARTS = list(range(0, 21))

MIN_SCORE = 0.55


def load_json_cached(path, cache):
    if path not in cache:
        cache[path] = json.loads(Path(path).read_text(encoding="utf-8"))
    return cache[path]


def get_frame(data, frame_id):
    frames = data.get("frames", {})
    return frames.get(str(frame_id))


def get_player(frame):
    if not frame:
        return None
    return frame.get("players", {}).get(PLAYER_ID)


def center_pos(player):
    pos = player.get("pos")
    if not pos:
        return None

    pts = []
    for idx in CENTER_PARTS:
        if idx < len(pos):
            pts.append(pos[idx])

    if not pts:
        return None

    x = sum(p[0] for p in pts) / len(pts)
    y = sum(p[1] for p in pts) / len(pts)
    z = sum(p[2] for p in pts) / len(pts)

    return x, y, z


def action_stats(action):
    leg_vals = [action[i] for i in LEG_IDXS if i < len(action)]
    leg_active = sum(1 for v in leg_vals if int(v) != 0)

    left = leg_vals[:5]
    right = leg_vals[5:]

    left_active = sum(1 for v in left if int(v) != 0)
    right_active = sum(1 for v in right if int(v) != 0)

    return leg_active, left_active, right_active


def score_motion(seq, data):
    start = int(seq["start_frame"])
    end = int(seq["end_frame"])
    mid = start + max(1, (end - start) // 2)

    f0 = get_frame(data, start)
    f1 = get_frame(data, mid)
    f2 = get_frame(data, end)

    p0 = get_player(f0)
    p1 = get_player(f1)
    p2 = get_player(f2)

    if not p0 or not p1 or not p2:
        return 0.0, None

    c0 = center_pos(p0)
    c1 = center_pos(p1)
    c2 = center_pos(p2)

    if not c0 or not c1 or not c2:
        return 0.0, None

    x0, y0, z0 = c0
    x1, y1, z1 = c1
    x2, y2, z2 = c2

    dy = y2 - y0
    dx = abs(x2 - x0)
    dz = z2 - z0
    z_drop = max(0.0, z0 - z2)
    z_range = max(z0, z1, z2) - min(z0, z1, z2)

    action = seq.get("action", [])
    leg_active, left_active, right_active = action_stats(action)

    score = 0.0

    # Avancer sur Y, sans exiger trop fort parce que certaines maps ont directions variées.
    if abs(dy) > 0.5:
        score += min(abs(dy) / 5.0, 1.0) * 0.30

    # Préférence marche: mouvement pas uniquement latéral.
    if abs(dy) > dx * 0.8:
        score += 0.15

    # Hauteur stable: pas de plongeon.
    if z_drop < 2.5:
        score += 0.20
    elif z_drop < 5.0:
        score += 0.08
    else:
        score -= 0.25

    if z_range < 6.0:
        score += 0.12
    elif z_range > 12.0:
        score -= 0.15

    # Jambes actives et bilatérales.
    if leg_active >= 3:
        score += 0.15
    if left_active > 0 and right_active > 0:
        score += 0.10

    # Pénalité drift latéral absurde.
    if dx > 8.0:
        score -= 0.15

    score = max(0.0, min(1.0, score))

    details = {
        "dy": round(dy, 4),
        "abs_dy": round(abs(dy), 4),
        "dx": round(dx, 4),
        "dz": round(dz, 4),
        "z_drop": round(z_drop, 4),
        "z_range": round(z_range, 4),
        "leg_active": leg_active,
        "left_active": left_active,
        "right_active": right_active,
    }

    return score, details


def main():
    cache = {}

    total = 0
    kept = 0
    rejected = 0
    errors = 0

    bins = Counter()
    mods = Counter()
    reasons = Counter()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with IN_PATH.open("r", encoding="utf-8") as f_in, OUT_PATH.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip():
                continue

            total += 1

            try:
                seq = json.loads(line)
                data = load_json_cached(seq["source_json"], cache)

                score, details = score_motion(seq, data)

                bin_key = f"{int(score * 10) / 10:.1f}"
                bins[bin_key] += 1

                if details is None:
                    rejected += 1
                    reasons["missing_motion"] += 1
                    continue

                if score < MIN_SCORE:
                    rejected += 1
                    reasons["low_score"] += 1
                    continue

                seq["walk_motion_score"] = round(score, 4)
                seq["walk_motion_details"] = details

                f_out.write(json.dumps(seq, ensure_ascii=False) + "\n")

                kept += 1
                mods[seq.get("mod", "UNKNOWN")] += 1

            except Exception as e:
                errors += 1
                rejected += 1
                reasons[type(e).__name__] += 1

    summary = {
        "input": str(IN_PATH),
        "output": str(OUT_PATH),
        "total": total,
        "kept": kept,
        "rejected": rejected,
        "errors": errors,
        "min_score": MIN_SCORE,
        "score_bins": dict(sorted(bins.items())),
        "top_mods": mods.most_common(20),
        "reasons": dict(reasons),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
