#!/usr/bin/env python3
"""
extract_early_walking_axis_v22.py

V22 = extraction marche early-replay axis-aware.

Pourquoi:
  V21 rejetait presque tout en reject_no_forward car il supposait que l'avant = X.
  Beaucoup de replays Toribash avancent en Y selon le mod/orientation.

Ce script:
  - scanne seulement le début des replays parkour
  - détecte automatiquement l'axe dominant X/Y au début
  - garde des fenêtres avec déplacement modéré sur cet axe
  - favorise tête/épaules stables + pieds bas
  - extrait les joint_pairs humains réels

Sorties:
  generated_replays/early_walking_axis_skills_v22.json
  generated_replays/early_walking_axis_skills_v22_summary.json
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
PARKOUR_JSON = ROOT / "datasets" / "parkour_json"
OUT_DIR = ROOT / "generated_replays"
OUT_SKILLS = OUT_DIR / "early_walking_axis_skills_v22.json"
OUT_SUMMARY = OUT_DIR / "early_walking_axis_skills_v22_summary.json"

PLAYER = "0"
MAX_REPLAYS = None
EARLY_MAX_FRAME = 260
WINDOW_ACTIONS_MIN = 4
WINDOW_ACTIONS_MAX = 10
TOP_PER_CATEGORY = 220
MAX_TOTAL_SKILLS = 700

# Indices POS Toribash approximatifs d'après nos observations:
# 0 head, 1 chest, 2 lumbar/stomach-ish, 3 groin, 4/7 pecs/shoulders-ish,
# 18/19 feet-ish dans plusieurs exports précédents.
HEAD = 0
CHEST = 1
CORE = 2
L_SHOULDER = 4
R_SHOULDER = 7
L_FOOT = 18
R_FOOT = 19
HANDISH = {10, 11, 12, 13}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def get_frames(data: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    raw = data.get("frames", {})
    if not isinstance(raw, dict):
        return []
    out = []
    for k, v in raw.items():
        try:
            out.append((int(k), v))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out


def player(frame: dict[str, Any]) -> dict[str, Any]:
    return frame.get("players", {}).get(PLAYER, {}) if isinstance(frame, dict) else {}


def pos_of(frame: dict[str, Any], idx: int) -> list[float] | None:
    p = player(frame).get("pos")
    if not isinstance(p, list) or idx >= len(p):
        return None
    q = p[idx]
    if not isinstance(q, list) or len(q) < 3:
        return None
    try:
        return [float(q[0]), float(q[1]), float(q[2])]
    except Exception:
        return None


def pairs_of(frame: dict[str, Any]) -> list[list[int]]:
    p = player(frame)
    pairs = p.get("joint_pairs")
    if isinstance(pairs, list):
        out = []
        for pair in pairs:
            if isinstance(pair, list) and len(pair) >= 2:
                try:
                    j = int(pair[0]); v = int(pair[1])
                    if 0 <= j <= 19 and 1 <= v <= 4:
                        out.append([j, v])
                except Exception:
                    pass
        return out
    joints = p.get("joints")
    if isinstance(joints, dict):
        out = []
        for k, v in joints.items():
            try:
                j = int(k); val = int(v)
                if 0 <= j <= 19 and 1 <= val <= 4:
                    out.append([j, val])
            except Exception:
                pass
        return sorted(out)
    return []


def dist2(a: list[float], b: list[float], axes=(0, 1)) -> float:
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in axes))


def zrange(vals: list[list[float]]) -> float:
    if not vals:
        return 999.0
    zs = [v[2] for v in vals]
    return max(zs) - min(zs)


def detect_axis(frames: list[tuple[int, dict[str, Any]]]) -> tuple[int, int]:
    early = [(n, f) for n, f in frames if 0 <= n <= EARLY_MAX_FRAME]
    if len(early) < 2:
        return 0, 1
    first = None; last = None
    for _, f in early:
        c = pos_of(f, CORE) or pos_of(f, CHEST) or pos_of(f, HEAD)
        if c and first is None:
            first = c
        if c:
            last = c
    if not first or not last:
        return 0, 1
    dx = last[0] - first[0]
    dy = last[1] - first[1]
    axis = 1 if abs(dy) > abs(dx) else 0
    sign = 1 if (dy if axis == 1 else dx) >= 0 else -1
    return axis, sign


def avg_pos(frames: list[dict[str, Any]], idx: int) -> list[float] | None:
    vals = [pos_of(f, idx) for f in frames]
    vals = [v for v in vals if v]
    if not vals:
        return None
    return [mean([v[i] for v in vals]) for i in range(3)]


def classify_window(replay_name: str, frames_slice: list[tuple[int, dict[str, Any]]], axis: int, sign: int) -> tuple[bool, str, dict[str, Any]]:
    nums = [n for n, _ in frames_slice]
    only = [f for _, f in frames_slice]
    if len(only) < 3:
        return False, "reject_short", {}

    f0 = only[0]; f1 = only[-1]
    c0 = pos_of(f0, CORE) or pos_of(f0, CHEST) or pos_of(f0, HEAD)
    c1 = pos_of(f1, CORE) or pos_of(f1, CHEST) or pos_of(f1, HEAD)
    if not c0 or not c1:
        return False, "missing_core_pos", {}

    forward = (c1[axis] - c0[axis]) * sign
    side_axis = 1 - axis
    sideways = abs(c1[side_axis] - c0[side_axis])

    # Early walking: avance modérée, pas glissade/trick énorme.
    if forward < 0.08:
        return False, "reject_no_forward_axis", {"forward": forward, "axis": axis}
    if forward > 5.5:
        return False, "reject_too_fast_axis", {"forward": forward, "axis": axis}
    if sideways > max(3.2, forward * 1.6):
        return False, "reject_sideways", {"forward": forward, "sideways": sideways}

    heads = [pos_of(f, HEAD) for f in only]
    heads = [h for h in heads if h]
    chest = [pos_of(f, CHEST) for f in only]
    chest = [h for h in chest if h]
    shoulders_l = [pos_of(f, L_SHOULDER) for f in only]
    shoulders_r = [pos_of(f, R_SHOULDER) for f in only]
    shoulders = [s for s in shoulders_l + shoulders_r if s]
    lf = [pos_of(f, L_FOOT) for f in only]
    rf = [pos_of(f, R_FOOT) for f in only]
    lf = [p for p in lf if p]; rf = [p for p in rf if p]
    if not heads or not shoulders or not lf or not rf:
        return False, "missing_body_pos", {}

    head_avg = mean(h[2] for h in heads)
    shoulder_avg = mean(s[2] for s in shoulders)
    feet_all = lf + rf
    foot_min = min(p[2] for p in feet_all)
    # Seuils relatifs au corps, pas absolus, car les mods ont des hauteurs différentes.
    body_height = max(0.1, head_avg - foot_min)
    if body_height < 2.5:
        return False, "reject_head_low_relative", {"body_height": body_height}
    if shoulder_avg - foot_min < 1.4:
        return False, "reject_shoulders_low_relative", {"shoulder_height": shoulder_avg - foot_min}
    if zrange(heads) > 4.5:
        return False, "reject_head_z_unstable", {"zrange": zrange(heads)}
    if zrange(shoulders) > 3.8:
        return False, "reject_shoulders_z_unstable", {"zrange": zrange(shoulders)}
    if zrange(feet_all) > 5.8:
        return False, "reject_feet_vertical_trick", {"zrange": zrange(feet_all)}

    # Pied proche du sol: on considère proche du minimum local du replay/window.
    near_thresh = foot_min + 0.55
    one_planted = 0
    both_air = 0
    planted_drifts = []
    swing_forwards = []
    for i in range(len(only)):
        l = pos_of(only[i], L_FOOT); r = pos_of(only[i], R_FOOT)
        if not l or not r:
            continue
        l_near = l[2] <= near_thresh
        r_near = r[2] <= near_thresh
        if l_near or r_near:
            one_planted += 1
        if not l_near and not r_near:
            both_air += 1
    n = max(1, len(only))
    if one_planted / n < 0.45:
        return False, "reject_no_ground_contact", {"ratio": one_planted / n}
    if both_air / n > 0.50:
        return False, "reject_too_much_air", {"ratio": both_air / n}

    # Swing foot: au moins un pied avance un peu plus que l'autre.
    l0, l1 = pos_of(f0, L_FOOT), pos_of(f1, L_FOOT)
    r0, r1 = pos_of(f0, R_FOOT), pos_of(f1, R_FOOT)
    if not l0 or not l1 or not r0 or not r1:
        return False, "missing_feet_endpoints", {}
    l_forward = (l1[axis] - l0[axis]) * sign
    r_forward = (r1[axis] - r0[axis]) * sign
    swing = max(l_forward, r_forward)
    planted = min(abs(l_forward), abs(r_forward))
    if swing < 0.05:
        return False, "reject_no_swing_forward", {"swing": swing}
    if swing > 7.5:
        return False, "reject_swing_too_big", {"swing": swing}
    if planted > 4.5:
        return False, "reject_no_planted_reference", {"planted": planted}

    action_frames = []
    actions = []
    for nframe, frame in frames_slice:
        pairs = pairs_of(frame)
        if pairs:
            action_frames.append(nframe)
            # dt en turns relatifs, conservé plus tard en export compact.
            actions.append({"frame": int(nframe), "pairs": pairs})
    if len(actions) < WINDOW_ACTIONS_MIN:
        return False, "reject_not_enough_actions", {"actions": len(actions)}
    if len(actions) > WINDOW_ACTIONS_MAX:
        # On compactera, mais trop de micro-actions peut être un trick/action spam.
        return False, "reject_too_many_action_frames", {"actions": len(actions)}
    avg_pairs = mean(len(a["pairs"]) for a in actions)
    peak_pairs = max(len(a["pairs"]) for a in actions)
    if avg_pairs > 8.5:
        return False, "reject_too_many_joints", {"avg_pairs": avg_pairs}
    if peak_pairs > 16:
        return False, "reject_joint_spike", {"peak_pairs": peak_pairs}

    # Catégorie gauche/droite selon pied qui avance le plus.
    category = "step_left" if l_forward > r_forward else "step_right"
    if forward > 1.2 and swing > 0.8:
        category = category + "_forward"
    elif forward < 0.55:
        category = category + "_micro"

    stability = 1.0 / (1.0 + zrange(heads) + zrange(shoulders) * 0.8)
    score = forward * 45.0 + swing * 18.0 + stability * 80.0 - sideways * 6.0 - both_air * 0.7

    sig_counter = Counter()
    for a in actions:
        for j, _ in a["pairs"]:
            sig_counter[j] += 1
    signature = "-".join(str(j) for j, _ in sig_counter.most_common(8))

    info = {
        "replay": replay_name,
        "start_frame": nums[0],
        "end_frame": nums[-1],
        "axis": "y" if axis == 1 else "x",
        "axis_index": axis,
        "axis_sign": sign,
        "forward": round(forward, 4),
        "sideways": round(sideways, 4),
        "left_forward": round(l_forward, 4),
        "right_forward": round(r_forward, 4),
        "swing_forward": round(swing, 4),
        "head_avg_z": round(head_avg, 4),
        "shoulder_avg_z": round(shoulder_avg, 4),
        "foot_min_z": round(foot_min, 4),
        "head_zrange": round(zrange(heads), 4),
        "shoulder_zrange": round(zrange(shoulders), 4),
        "feet_zrange": round(zrange(feet_all), 4),
        "one_planted_ratio": round(one_planted / n, 4),
        "both_air_ratio": round(both_air / n, 4),
        "action_count": len(actions),
        "avg_pairs": round(avg_pairs, 3),
        "peak_pairs": peak_pairs,
        "score": round(score, 4),
        "signature": signature,
        "category": category,
        "actions_abs": actions,
    }
    return True, category, info


def compact_actions(actions_abs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not actions_abs:
        return []
    base = min(int(a["frame"]) for a in actions_abs)
    out = []
    last_dt = -999
    for a in actions_abs:
        frame = int(a["frame"])
        # Convertit en pas de 5 frames depuis le début du skill.
        dt = max(0, round((frame - base) / 5) * 5)
        if dt == last_dt:
            # fusion simple si plusieurs actions même dt
            prev = {j: v for j, v in out[-1]["pairs"]}
            for j, v in a["pairs"]:
                prev[int(j)] = int(v)
            out[-1]["pairs"] = [[j, prev[j]] for j in sorted(prev)]
        else:
            out.append({"dt": dt, "pairs": [[int(j), int(v)] for j, v in a["pairs"]]})
        last_dt = dt
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(PARKOUR_JSON.glob("*.json"))
    if MAX_REPLAYS:
        paths = paths[:MAX_REPLAYS]
    print("Parkour JSON:", PARKOUR_JSON)
    print("Replays:", len(paths))

    raw_skills = []
    rejects = Counter()
    axis_counts = Counter()
    by_replay = Counter()
    sample_rejects = defaultdict(list)

    skill_id = 0
    for pi, path in enumerate(paths, 1):
        data = load_json(path)
        if not data:
            rejects["bad_json"] += 1
            continue
        frames = get_frames(data)
        early = [(n, f) for n, f in frames if 0 <= n <= EARLY_MAX_FRAME]
        if len(early) < 8:
            rejects["too_short_early"] += 1
            continue
        axis, sign = detect_axis(early)
        axis_counts[("y" if axis == 1 else "x", sign)] += 1

        # Fenêtres early seulement, volontairement chevauchantes.
        for win_len in (6, 8, 10, 12):
            step = max(2, win_len // 3)
            for start in range(0, max(0, len(early) - win_len + 1), step):
                sl = early[start:start+win_len]
                ok, cat, info = classify_window(path.name, sl, axis, sign)
                if not ok:
                    rejects[cat] += 1
                    if len(sample_rejects[cat]) < 3:
                        sample_rejects[cat].append(info)
                    continue
                actions = compact_actions(info.pop("actions_abs"))
                if len(actions) < 2:
                    rejects["reject_empty_after_compact"] += 1
                    continue
                skill = {
                    "id": skill_id,
                    "name": f"v22_{cat}_{skill_id:04d}",
                    "version": 22,
                    "source": "early_axis_walk",
                    "category": cat,
                    "score": info["score"],
                    "length": len(actions),
                    "actions": actions,
                    **info,
                }
                raw_skills.append(skill)
                by_replay[path.name] += 1
                skill_id += 1

        if pi % 25 == 0 or pi == len(paths):
            print(f"  {pi}/{len(paths)} replays | accepted={len(raw_skills)}")

    # Dédup signature + catégorie + source approximative.
    raw_skills.sort(key=lambda s: s["score"], reverse=True)
    final = []
    seen = set()
    cat_counts = Counter()
    for s in raw_skills:
        key = (s["category"], s["signature"], round(float(s["forward"]), 1), round(float(s["swing_forward"]), 1))
        if key in seen:
            continue
        seen.add(key)
        if cat_counts[s["category"]] >= TOP_PER_CATEGORY:
            continue
        final.append(s)
        cat_counts[s["category"]] += 1
        if len(final) >= MAX_TOTAL_SKILLS:
            break

    data_out = {
        "name": "early_walking_axis_skills_v22",
        "version": 22,
        "description": "Early replay, axis-aware walking-step skills from real parsed Toribash RPL files.",
        "early_max_frame": EARLY_MAX_FRAME,
        "skills": final,
    }
    summary = {
        "version": 22,
        "input_replays": len(paths),
        "raw_accepted": len(raw_skills),
        "final_skill_count": len(final),
        "categories": dict(Counter(s["category"] for s in final)),
        "rejects": dict(rejects.most_common()),
        "axis_counts": {f"{a}:{sgn}": c for (a, sgn), c in axis_counts.items()},
        "accepted_replays": len(by_replay),
        "top_skills": [
            {k: s[k] for k in ["id", "category", "score", "forward", "sideways", "swing_forward", "head_zrange", "shoulder_zrange", "signature", "replay", "start_frame", "end_frame", "axis", "axis_sign"]}
            for s in final[:30]
        ],
        "sample_rejects": {k: v for k, v in list(sample_rejects.items())[:20]},
    }
    OUT_SKILLS.write_text(json.dumps(data_out, indent=2), encoding="utf-8")
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:", OUT_SKILLS)
    print("Summary:", OUT_SUMMARY)
    print("Categories:", summary["categories"])
    print("Axis counts:", summary["axis_counts"])
    print("Rejects top:", rejects.most_common(12))
    print("\nTop skills:")
    for s in final[:20]:
        print(
            f"  id={s['id']:4d} cat={s['category']:<20} score={s['score']:7.2f} "
            f"fwd={s['forward']:5.2f} side={s['sideways']:5.2f} swing={s['swing_forward']:5.2f} "
            f"axis={s['axis']}{s['axis_sign']} frames={s['start_frame']}-{s['end_frame']} sig={s['signature']}"
        )


if __name__ == "__main__":
    main()
