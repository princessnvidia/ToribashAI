#!/usr/bin/env python3
"""
extract_strict_walking_gait_v21.py

V21 = extraction stricte de vraie marche depuis les RPL parsés.

But:
  Ne plus extraire les tricks/parkour explosifs.
  Garder uniquement les fenêtres qui ressemblent biomécaniquement à un pas:
    - pieds proches du sol
    - au moins un pied planté/stable
    - l'autre pied avance modérément
    - épaules stables
    - tête/torse assez hauts
    - peu de variation verticale
    - déplacement horizontal positif mais non explosif

Entrée:
  datasets/parkour_json/*.json

Sorties:
  generated_replays/walking_gait_strict_skills_v21.json
  generated_replays/walking_gait_strict_skills_v21_summary.json
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET_DIR = ROOT / "datasets" / "parkour_json"
OUT_DIR = ROOT / "generated_replays"
OUT_SKILLS = OUT_DIR / "walking_gait_strict_skills_v21.json"
OUT_SUMMARY = OUT_DIR / "walking_gait_strict_skills_v21_summary.json"

# Indices de bodyparts d'après la structure Toribash classique observée.
# On garde plusieurs indices possibles pour rendre le filtre robuste.
HEAD_IDX = 0
CHEST_IDX = 1
L_SHOULDER_IDX = 5
R_SHOULDER_IDX = 8
L_FOOT_IDX = 19
R_FOOT_IDX = 20
L_HAND_IDX = 11
R_HAND_IDX = 12

# Fenêtres: courte marche/pas, pas un trick long.
WINDOW_SIZES = [6, 8, 10, 12]       # nombre de frames ACTION, pas frames simulation
MAX_SKILLS = 320
MAX_PER_REPLAY = 4
MAX_PER_CATEGORY = 180

# Seuils stricts. Ils sont volontairement conservateurs.
MIN_ACTION_FRAMES = 5
MIN_FORWARD_DX = 0.20
MAX_FORWARD_DX = 4.25               # rejette les lancés/tricks très rapides
MAX_SIDE_DY = 2.75
MAX_VERTICAL_RANGE = 4.25
MAX_HEAD_DROP = 3.25
MIN_HEAD_ABOVE_FEET = 8.0
MIN_SHOULDER_ABOVE_FEET = 5.5
MAX_SHOULDER_Z_RANGE = 2.25
MAX_SHOULDER_Y_RANGE = 3.0
MAX_SHOULDER_X_JITTER = 2.5
MAX_FOOT_Z_RANGE = 3.2
MAX_BOTH_FEET_AIR_RATIO = 0.20
MAX_HAND_LOW_RATIO = 0.22
MIN_ONE_FOOT_PLANTED_RATIO = 0.35
MIN_SWING_FOOT_ADVANCE = 0.18
MAX_SWING_FOOT_ADVANCE = 4.5
MAX_PLANTED_FOOT_DRIFT = 1.85
MAX_ACTIVE_PER_ACTION_AVG = 7.0
MAX_ACTIVE_PER_ACTION_PEAK = 12

# Dans les replays, le sol varie selon les mods. On estime le sol par le min Z des pieds.
GROUND_MARGIN = 1.10
PLANTED_MOVE_EPS = 0.95

# Joints à garder pour la marche. On enlève une partie des bras explosifs, mais on garde épaules/pecs utiles.
ALLOWED_JOINTS = set(range(20))


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sorted_frame_items(data: dict[str, Any]):
    frames = data.get("frames", {})
    if isinstance(frames, dict):
        items = []
        for k, v in frames.items():
            try:
                items.append((int(k), v))
            except Exception:
                continue
        return sorted(items, key=lambda x: x[0])
    if isinstance(frames, list):
        out = []
        for i, f in enumerate(frames):
            frame_no = f.get("frame", i) if isinstance(f, dict) else i
            try:
                frame_no = int(frame_no)
            except Exception:
                frame_no = i
            out.append((frame_no, f))
        return out
    return []


def p0(frame: dict[str, Any]) -> dict[str, Any]:
    return frame.get("players", {}).get("0", {}) if isinstance(frame, dict) else {}


def vec3(p):
    if not isinstance(p, list) or len(p) < 3:
        return None
    try:
        return [float(p[0]), float(p[1]), float(p[2])]
    except Exception:
        return None


def get_pos(frame: dict[str, Any], idx: int):
    pos = p0(frame).get("pos")
    if not isinstance(pos, list) or idx >= len(pos):
        return None
    return vec3(pos[idx])


def dist_xy(a, b) -> float:
    if a is None or b is None:
        return 999.0
    return math.hypot(a[0] - b[0], a[1] - b[1])


def get_joint_pairs(frame: dict[str, Any]):
    pairs = p0(frame).get("joint_pairs")
    if isinstance(pairs, list):
        out = []
        for pair in pairs:
            if isinstance(pair, list) and len(pair) >= 2:
                try:
                    j = int(pair[0])
                    v = int(pair[1])
                except Exception:
                    continue
                if 0 <= j < 20 and 1 <= v <= 4 and j in ALLOWED_JOINTS:
                    out.append([j, v])
        return out

    joints = p0(frame).get("joints")
    if isinstance(joints, dict):
        out = []
        for k, v in joints.items():
            try:
                j = int(k)
                val = int(v)
            except Exception:
                continue
            if 0 <= j < 20 and 1 <= val <= 4 and j in ALLOWED_JOINTS:
                out.append([j, val])
        return sorted(out)
    return []


def avg(values):
    vals = [v for v in values if v is not None]
    return mean(vals) if vals else None


def z_range(points):
    zs = [p[2] for p in points if p is not None]
    return max(zs) - min(zs) if len(zs) >= 2 else 999.0


def infer_ground_z(window_frames) -> float | None:
    zs = []
    for _, f in window_frames:
        lf = get_pos(f, L_FOOT_IDX)
        rf = get_pos(f, R_FOOT_IDX)
        if lf: zs.append(lf[2])
        if rf: zs.append(rf[2])
    if not zs:
        return None
    zs.sort()
    # Robust: proche du min, mais pas uniquement un outlier.
    return zs[min(2, len(zs) - 1)]


def classify_window(window_frames) -> tuple[bool, str, dict[str, Any]]:
    first_no, first = window_frames[0]
    last_no, last = window_frames[-1]

    chest0 = get_pos(first, CHEST_IDX) or get_pos(first, HEAD_IDX)
    chest1 = get_pos(last, CHEST_IDX) or get_pos(last, HEAD_IDX)
    head0 = get_pos(first, HEAD_IDX)
    head1 = get_pos(last, HEAD_IDX)

    if chest0 is None or chest1 is None or head0 is None or head1 is None:
        return False, "missing_core_pos", {}

    dx = chest1[0] - chest0[0]
    dy = chest1[1] - chest0[1]

    if dx < MIN_FORWARD_DX:
        return False, "reject_no_forward", {"dx": dx, "dy": dy}
    if dx > MAX_FORWARD_DX:
        return False, "reject_too_fast", {"dx": dx, "dy": dy}
    if abs(dy) > MAX_SIDE_DY:
        return False, "reject_sideways", {"dx": dx, "dy": dy}

    ground = infer_ground_z(window_frames)
    if ground is None:
        return False, "missing_feet", {}

    heads = []
    chests = []
    shoulders_mid = []
    feet_l = []
    feet_r = []
    hands_low = 0
    both_air = 0
    one_planted = 0
    action_counts = []

    for _, f in window_frames:
        h = get_pos(f, HEAD_IDX)
        c = get_pos(f, CHEST_IDX)
        ls = get_pos(f, L_SHOULDER_IDX)
        rs = get_pos(f, R_SHOULDER_IDX)
        lf = get_pos(f, L_FOOT_IDX)
        rf = get_pos(f, R_FOOT_IDX)
        lh = get_pos(f, L_HAND_IDX)
        rh = get_pos(f, R_HAND_IDX)

        if h: heads.append(h)
        if c: chests.append(c)
        if lf: feet_l.append(lf)
        if rf: feet_r.append(rf)
        if ls and rs:
            shoulders_mid.append([(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2, (ls[2] + rs[2]) / 2])

        if lf and rf:
            lf_near = lf[2] <= ground + GROUND_MARGIN
            rf_near = rf[2] <= ground + GROUND_MARGIN
            if not lf_near and not rf_near:
                both_air += 1
            if lf_near or rf_near:
                one_planted += 1

        if lh and lh[2] <= ground + 2.0:
            hands_low += 1
        if rh and rh[2] <= ground + 2.0:
            hands_low += 1

        pairs = get_joint_pairs(f)
        if pairs:
            action_counts.append(len(pairs))

    n = len(window_frames)
    if len(heads) < max(3, n // 2) or len(feet_l) < max(3, n // 2) or len(feet_r) < max(3, n // 2):
        return False, "missing_enough_pos", {}

    head_min = min(p[2] for p in heads)
    head_avg = mean(p[2] for p in heads)
    foot_min = min([p[2] for p in feet_l + feet_r])
    shoulder_avg_z = avg([p[2] for p in shoulders_mid]) or 0.0

    if head_avg - foot_min < MIN_HEAD_ABOVE_FEET:
        return False, "reject_head_low", {"dx": dx, "head_avg": head_avg, "foot_min": foot_min}
    if shoulder_avg_z - foot_min < MIN_SHOULDER_ABOVE_FEET:
        return False, "reject_shoulders_low", {"dx": dx, "shoulder_avg_z": shoulder_avg_z}
    if head0[2] - head1[2] > MAX_HEAD_DROP:
        return False, "reject_head_drop", {"dx": dx, "head_drop": head0[2] - head1[2]}

    vertical_all = heads + chests + shoulders_mid
    if z_range(vertical_all) > MAX_VERTICAL_RANGE:
        return False, "reject_vertical_explosive", {"dx": dx, "zrange": z_range(vertical_all)}
    if z_range(shoulders_mid) > MAX_SHOULDER_Z_RANGE:
        return False, "reject_shoulders_unstable_z", {"dx": dx, "shoulder_zrange": z_range(shoulders_mid)}

    if shoulders_mid:
        sx = [p[0] for p in shoulders_mid]
        sy = [p[1] for p in shoulders_mid]
        if max(sx) - min(sx) > MAX_SHOULDER_X_JITTER:
            return False, "reject_shoulders_x_jitter", {"dx": dx, "shoulder_xrange": max(sx) - min(sx)}
        if max(sy) - min(sy) > MAX_SHOULDER_Y_RANGE:
            return False, "reject_shoulders_y_jitter", {"dx": dx, "shoulder_yrange": max(sy) - min(sy)}

    if z_range(feet_l + feet_r) > MAX_FOOT_Z_RANGE:
        return False, "reject_feet_vertical_trick", {"dx": dx, "foot_zrange": z_range(feet_l + feet_r)}

    if both_air / max(1, n) > MAX_BOTH_FEET_AIR_RATIO:
        return False, "reject_feet_air", {"dx": dx, "both_air_ratio": both_air / max(1, n)}
    if one_planted / max(1, n) < MIN_ONE_FOOT_PLANTED_RATIO:
        return False, "reject_no_planted_foot", {"dx": dx, "one_planted_ratio": one_planted / max(1, n)}
    if hands_low / max(1, 2 * n) > MAX_HAND_LOW_RATIO:
        return False, "reject_hands_low", {"dx": dx, "hands_low_ratio": hands_low / max(1, 2 * n)}

    # Analyse pied planté / pied swing.
    l_start, l_end = feet_l[0], feet_l[-1]
    r_start, r_end = feet_r[0], feet_r[-1]
    l_move = dist_xy(l_start, l_end)
    r_move = dist_xy(r_start, r_end)
    l_dx = l_end[0] - l_start[0]
    r_dx = r_end[0] - r_start[0]

    planted_side = None
    swing_dx = None
    planted_drift = None
    if l_move <= r_move:
        planted_side = "left"
        planted_drift = l_move
        swing_dx = r_dx
    else:
        planted_side = "right"
        planted_drift = r_move
        swing_dx = l_dx

    if planted_drift > MAX_PLANTED_FOOT_DRIFT:
        return False, "reject_no_stable_foot", {"dx": dx, "planted_drift": planted_drift}
    if swing_dx < MIN_SWING_FOOT_ADVANCE:
        return False, "reject_no_swing_forward", {"dx": dx, "swing_dx": swing_dx}
    if swing_dx > MAX_SWING_FOOT_ADVANCE:
        return False, "reject_swing_too_big", {"dx": dx, "swing_dx": swing_dx}

    if len(action_counts) < MIN_ACTION_FRAMES:
        return False, "reject_not_enough_actions", {"dx": dx, "action_frames": len(action_counts)}
    if mean(action_counts) > MAX_ACTIVE_PER_ACTION_AVG:
        return False, "reject_too_many_joints", {"dx": dx, "avg_actions": mean(action_counts)}
    if max(action_counts) > MAX_ACTIVE_PER_ACTION_PEAK:
        return False, "reject_joint_spike", {"dx": dx, "peak_actions": max(action_counts)}

    # Score marche: pas dx max, mais stabilité + progression modérée.
    shoulder_zr = z_range(shoulders_mid)
    vertical_r = z_range(vertical_all)
    score = 0.0
    score += dx * 35.0
    score += min(3.0, swing_dx) * 20.0
    score += max(0.0, 3.0 - abs(dy)) * 8.0
    score += max(0.0, 3.0 - shoulder_zr) * 18.0
    score += max(0.0, 5.0 - vertical_r) * 10.0
    score += max(0.0, 2.0 - planted_drift) * 16.0
    score += max(0.0, head_avg - foot_min - 8.0) * 2.0
    score -= max(0.0, mean(action_counts) - 4.0) * 6.0

    category = "step_left" if planted_side == "right" else "step_right"
    if dx < 0.65:
        category = "micro_step"
    if head_avg - foot_min > 18.0 and shoulder_zr < 1.3 and dx < 1.25:
        category = "upright_step"

    metrics = {
        "dx": round(dx, 4),
        "dy": round(dy, 4),
        "head_avg": round(head_avg, 4),
        "head_min": round(head_min, 4),
        "foot_min": round(foot_min, 4),
        "head_above_feet": round(head_avg - foot_min, 4),
        "shoulder_avg_z": round(shoulder_avg_z, 4),
        "shoulder_z_range": round(shoulder_zr, 4),
        "vertical_range": round(vertical_r, 4),
        "foot_z_range": round(z_range(feet_l + feet_r), 4),
        "both_air_ratio": round(both_air / max(1, n), 4),
        "one_planted_ratio": round(one_planted / max(1, n), 4),
        "hands_low_ratio": round(hands_low / max(1, 2 * n), 4),
        "planted_side": planted_side,
        "planted_drift": round(planted_drift, 4),
        "swing_dx": round(swing_dx, 4),
        "left_foot_dx": round(l_dx, 4),
        "right_foot_dx": round(r_dx, 4),
        "avg_action_count": round(mean(action_counts), 4),
        "peak_action_count": max(action_counts),
        "score": round(score, 4),
    }
    return True, category, metrics


def compact_actions(window_frames):
    actions = []
    first_frame = window_frames[0][0]
    for frame_no, f in window_frames:
        pairs = get_joint_pairs(f)
        if not pairs:
            continue
        # Dedup joint in frame, keep last value.
        clean = {}
        for j, v in pairs:
            clean[int(j)] = int(v)
        actions.append({
            "dt": int(frame_no - first_frame),
            "pairs": [[j, clean[j]] for j in sorted(clean)],
        })
    return actions


def signature(actions):
    c = Counter()
    for a in actions:
        for j, _ in a.get("pairs", []):
            c[j] += 1
    return "-".join(str(j) for j, _ in c.most_common(10))


def main() -> None:
    paths = sorted(DATASET_DIR.glob("*.json"))
    print("Parkour JSON:", DATASET_DIR)
    print("Replays:", len(paths))

    raw_skills = []
    rejects = Counter()
    accepted_by_replay = Counter()
    seen_sigs = set()

    for pi, path in enumerate(paths, 1):
        try:
            data = load_json(path)
        except Exception as e:
            rejects["bad_json"] += 1
            continue

        items = sorted_frame_items(data)
        # Ne garder que frames avec position exploitable.
        if len(items) < 12:
            rejects["too_short_replay"] += 1
            continue

        replay_added = 0
        for w in WINDOW_SIZES:
            if len(items) < w:
                continue
            # Stride volontaire pour diversité.
            stride = max(1, w // 3)
            for start in range(0, len(items) - w + 1, stride):
                window = items[start:start + w]
                ok, cat_or_reason, metrics = classify_window(window)
                if not ok:
                    rejects[cat_or_reason] += 1
                    continue

                actions = compact_actions(window)
                if len(actions) < MIN_ACTION_FRAMES:
                    rejects["reject_actions_after_compact"] += 1
                    continue
                sig = signature(actions)
                if not sig:
                    rejects["empty_signature"] += 1
                    continue

                # Dédup agressif: même replay + catégorie + signature trop similaire.
                dedup_key = (path.name, cat_or_reason, sig, round(metrics["dx"], 1), round(metrics["swing_dx"], 1))
                if dedup_key in seen_sigs:
                    rejects["reject_duplicate"] += 1
                    continue
                seen_sigs.add(dedup_key)

                skill = {
                    "id": None,
                    "version": "21",
                    "category": cat_or_reason,
                    "source_replay": str(path),
                    "source_file": path.name,
                    "start_frame": int(window[0][0]),
                    "end_frame": int(window[-1][0]),
                    "length": len(actions),
                    "signature": sig,
                    "score": metrics["score"],
                    "metrics": metrics,
                    "actions": actions,
                }
                raw_skills.append(skill)
                replay_added += 1
                accepted_by_replay[path.name] += 1

                if replay_added >= MAX_PER_REPLAY:
                    break
            if replay_added >= MAX_PER_REPLAY:
                break

        if pi % 25 == 0 or pi == len(paths):
            print(f"  {pi}/{len(paths)} replays | accepted={len(raw_skills)}")

    # Trier par category puis score, cap par catégorie.
    by_cat = defaultdict(list)
    for s in raw_skills:
        by_cat[s["category"]].append(s)

    final = []
    for cat, arr in by_cat.items():
        arr.sort(key=lambda s: s["score"], reverse=True)
        final.extend(arr[:MAX_PER_CATEGORY])

    final.sort(key=lambda s: s["score"], reverse=True)
    final = final[:MAX_SKILLS]
    for i, s in enumerate(final):
        s["id"] = i

    out = {
        "name": "walking_gait_strict_skills_v21",
        "version": 21,
        "description": "Strict foot-contact / stable-shoulder walking gait skills extracted from real parsed RPL replays.",
        "bodypart_indices": {
            "head": HEAD_IDX,
            "chest": CHEST_IDX,
            "left_shoulder": L_SHOULDER_IDX,
            "right_shoulder": R_SHOULDER_IDX,
            "left_foot": L_FOOT_IDX,
            "right_foot": R_FOOT_IDX,
            "left_hand": L_HAND_IDX,
            "right_hand": R_HAND_IDX,
        },
        "thresholds": {
            "min_forward_dx": MIN_FORWARD_DX,
            "max_forward_dx": MAX_FORWARD_DX,
            "max_side_dy": MAX_SIDE_DY,
            "max_vertical_range": MAX_VERTICAL_RANGE,
            "max_head_drop": MAX_HEAD_DROP,
            "min_head_above_feet": MIN_HEAD_ABOVE_FEET,
            "max_shoulder_z_range": MAX_SHOULDER_Z_RANGE,
            "max_both_feet_air_ratio": MAX_BOTH_FEET_AIR_RATIO,
            "min_one_foot_planted_ratio": MIN_ONE_FOOT_PLANTED_RATIO,
        },
        "skill_count": len(final),
        "skills": final,
    }

    summary = {
        "version": 21,
        "input_replays": len(paths),
        "raw_accepted": len(raw_skills),
        "final_skill_count": len(final),
        "categories": Counter(s["category"] for s in final),
        "rejects": rejects,
        "top_skills": [
            {
                "id": s["id"],
                "category": s["category"],
                "score": s["score"],
                "dx": s["metrics"]["dx"],
                "swing_dx": s["metrics"]["swing_dx"],
                "head_above_feet": s["metrics"]["head_above_feet"],
                "shoulder_z_range": s["metrics"]["shoulder_z_range"],
                "vertical_range": s["metrics"]["vertical_range"],
                "planted_side": s["metrics"]["planted_side"],
                "source_file": s["source_file"],
                "frames": f"{s['start_frame']}-{s['end_frame']}",
                "signature": s["signature"],
            }
            for s in final[:40]
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SKILLS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    # Convert counters to dicts for JSON.
    summary["categories"] = dict(summary["categories"])
    summary["rejects"] = dict(summary["rejects"])
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:", OUT_SKILLS)
    print("Summary:", OUT_SUMMARY)
    print("Categories:", summary["categories"])
    print("Rejects top:", rejects.most_common(14))
    print("\nTop skills:")
    for s in final[:20]:
        m = s["metrics"]
        print(
            f"  id={s['id']:4d} cat={s['category']:<12} score={s['score']:8.2f} "
            f"dx={m['dx']:5.2f} swing={m['swing_dx']:5.2f} "
            f"head+={m['head_above_feet']:5.2f} shZ={m['shoulder_z_range']:4.2f} "
            f"vR={m['vertical_range']:4.2f} plant={m['planted_side']:<5} "
            f"frames={s['start_frame']}-{s['end_frame']} sig={s['signature']}"
        )


if __name__ == "__main__":
    main()
