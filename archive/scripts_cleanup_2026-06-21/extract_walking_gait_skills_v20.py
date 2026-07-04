#!/usr/bin/env python3
"""
extract_walking_gait_skills_v20.py

V20 = extraction de marche par signature de pieds/contact.
On ne cherche plus juste dx>0 : on cherche des fenêtres où un pied reste planté
pendant que l'autre avance, puis on classe left_step/right_step/recover/stand.

Entrée:
  datasets/parkour_json/*.json

Sorties:
  generated_replays/parkour_walking_gait_skills_v20.json
  generated_replays/parkour_walking_gait_skills_v20_summary.json
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
OUT_SKILLS = OUT_DIR / "parkour_walking_gait_skills_v20.json"
OUT_SUMMARY = OUT_DIR / "parkour_walking_gait_skills_v20_summary.json"

# Indices probables dans les POS Toribash parsés: les deux derniers points sont les pieds.
LEFT_FOOT_IDX = 19
RIGHT_FOOT_IDX = 20
HEAD_IDX = 0
CHEST_IDX = 1
HIP_IDX = 2

WINDOW_ACTIONS_MIN = 4
WINDOW_ACTIONS_MAX = 9
MAX_KEEP_PER_CATEGORY = 180

# Filtres gait. Relativement larges parce que les replays parkour sont sales.
MIN_FORWARD = 0.15
MAX_FORWARD = 5.50
MAX_SIDE = 3.80
MIN_HEAD_Z = 7.0
MAX_HEAD_Z_RANGE = 10.0
MAX_FOOT_PLANTED_MOVE = 0.85
MIN_SWING_MOVE = 0.45
MIN_SWING_OVER_PLANT_RATIO = 1.45
MAX_TOTAL_JOINTS_PER_ACTION = 10


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sorted_frame_items(data: dict[str, Any]):
    frames = data.get("frames", {})
    if not isinstance(frames, dict):
        return []
    out = []
    for k, frame in frames.items():
        try:
            out.append((int(k), frame))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out


def p0(frame: dict[str, Any]) -> dict[str, Any]:
    return frame.get("players", {}).get("0", {}) if isinstance(frame, dict) else {}


def get_pos(frame: dict[str, Any], idx: int) -> list[float] | None:
    pos = p0(frame).get("pos")
    if not isinstance(pos, list) or idx >= len(pos):
        return None
    try:
        return [float(pos[idx][0]), float(pos[idx][1]), float(pos[idx][2])]
    except Exception:
        return None


def dist_xy(a: list[float], b: list[float]) -> float:
    return math.hypot(float(b[0]) - float(a[0]), float(b[1]) - float(a[1]))


def get_pairs(frame: dict[str, Any]) -> list[list[int]]:
    pairs = p0(frame).get("joint_pairs")
    if not isinstance(pairs, list):
        return []
    clean = []
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) < 2:
            continue
        try:
            j = int(pair[0])
            v = int(pair[1])
        except Exception:
            continue
        if 0 <= j <= 19 and 1 <= v <= 4:
            clean.append([j, v])
    return clean


def frame_features(frame: dict[str, Any]) -> dict[str, Any] | None:
    lf = get_pos(frame, LEFT_FOOT_IDX)
    rf = get_pos(frame, RIGHT_FOOT_IDX)
    head = get_pos(frame, HEAD_IDX)
    hip = get_pos(frame, HIP_IDX) or get_pos(frame, CHEST_IDX)
    if lf is None or rf is None or head is None or hip is None:
        return None
    midx = (lf[0] + rf[0]) * 0.5
    midy = (lf[1] + rf[1]) * 0.5
    return {
        "lf": lf,
        "rf": rf,
        "head": head,
        "hip": hip,
        "mid": [midx, midy, (lf[2] + rf[2]) * 0.5],
    }


def window_motion(frames_by_no: dict[int, dict[str, Any]], start_no: int, end_no: int) -> dict[str, Any] | None:
    a = frames_by_no.get(start_no)
    b = frames_by_no.get(end_no)
    if a is None or b is None:
        return None
    fa = frame_features(a)
    fb = frame_features(b)
    if fa is None or fb is None:
        return None

    lf_move = dist_xy(fa["lf"], fb["lf"])
    rf_move = dist_xy(fa["rf"], fb["rf"])
    dx = fb["mid"][0] - fa["mid"][0]
    dy = fb["mid"][1] - fa["mid"][1]
    head_z0 = fa["head"][2]
    head_z1 = fb["head"][2]
    hip_z0 = fa["hip"][2]
    hip_z1 = fb["hip"][2]

    return {
        "dx": dx,
        "dy": dy,
        "lf_move": lf_move,
        "rf_move": rf_move,
        "head_z0": head_z0,
        "head_z1": head_z1,
        "head_z_min": min(head_z0, head_z1),
        "head_z_range": abs(head_z1 - head_z0),
        "hip_z0": hip_z0,
        "hip_z1": hip_z1,
        "hip_z_min": min(hip_z0, hip_z1),
    }


def classify_gait(m: dict[str, Any]) -> str | None:
    dx = m["dx"]
    dy = abs(m["dy"])
    lf = m["lf_move"]
    rf = m["rf_move"]
    head_min = m["head_z_min"]
    head_range = m["head_z_range"]

    if head_min < MIN_HEAD_Z or head_range > MAX_HEAD_Z_RANGE:
        return None
    if dy > MAX_SIDE:
        return None

    # Stand: presque pas de déplacement, tête haute/stable.
    if abs(dx) < 0.12 and dy < 0.35 and lf < 0.35 and rf < 0.35:
        return "stand"

    if dx < MIN_FORWARD or dx > MAX_FORWARD:
        return None

    left_planted = lf <= MAX_FOOT_PLANTED_MOVE and rf >= MIN_SWING_MOVE and rf >= lf * MIN_SWING_OVER_PLANT_RATIO
    right_planted = rf <= MAX_FOOT_PLANTED_MOVE and lf >= MIN_SWING_MOVE and lf >= rf * MIN_SWING_OVER_PLANT_RATIO

    if left_planted:
        return "right_step"   # pied gauche planté, pied droit avance
    if right_planted:
        return "left_step"    # pied droit planté, pied gauche avance

    # Les deux pieds bougent modérément: possible phase de transfert/lean.
    if 0.15 <= dx <= 2.2 and lf < 1.35 and rf < 1.35:
        return "lean_forward"

    # Récupération verticale sans gros twist latéral.
    if dx >= -0.15 and m["head_z1"] > m["head_z0"] + 0.4:
        return "recover_upright"

    return None


def signature(actions: list[dict[str, Any]]) -> str:
    c = Counter()
    for a in actions:
        for j, _v in a.get("pairs", []):
            c[int(j)] += 1
    return "-".join(str(j) for j, _ in c.most_common(8))


def score_skill(category: str, m: dict[str, Any], actions: list[dict[str, Any]]) -> float:
    avg_pairs = mean([len(a.get("pairs", [])) for a in actions]) if actions else 0
    dx = m["dx"]
    score = 0.0
    if category in {"left_step", "right_step"}:
        swing = max(m["lf_move"], m["rf_move"])
        plant = min(m["lf_move"], m["rf_move"])
        score += 120 + dx * 22 + swing * 12 - plant * 18
    elif category == "lean_forward":
        score += 80 + dx * 18
    elif category == "stand":
        score += 90 - abs(dx) * 80 - m["head_z_range"] * 4
    elif category == "recover_upright":
        score += 70 + (m["head_z1"] - m["head_z0"]) * 18 + max(0, dx) * 10
    score += min(30, m["head_z_min"] * 1.2)
    score -= max(0, avg_pairs - 6) * 6
    score -= abs(m["dy"]) * 4
    return round(score, 4)


def extract_from_replay(path: Path, next_id: int) -> tuple[list[dict[str, Any]], int]:
    data = load_json(path)
    if not data:
        return [], next_id
    items = sorted_frame_items(data)
    if len(items) < 8:
        return [], next_id

    frames_by_no = {no: frame for no, frame in items}
    action_frames = [(no, get_pairs(frame)) for no, frame in items if get_pairs(frame)]
    if len(action_frames) < WINDOW_ACTIONS_MIN:
        return [], next_id

    found: list[dict[str, Any]] = []
    seen_local = set()

    for i in range(0, len(action_frames) - WINDOW_ACTIONS_MIN + 1):
        for w in range(WINDOW_ACTIONS_MIN, WINDOW_ACTIONS_MAX + 1):
            if i + w > len(action_frames):
                continue
            chunk = action_frames[i:i + w]
            start_no = chunk[0][0]
            end_no = chunk[-1][0]
            if end_no <= start_no:
                continue
            m = window_motion(frames_by_no, start_no, end_no)
            if not m:
                continue

            actions = []
            too_big = False
            for frame_no, pairs in chunk:
                if len(pairs) > MAX_TOTAL_JOINTS_PER_ACTION:
                    too_big = True
                    break
                actions.append({"dt": int(frame_no - start_no), "pairs": pairs})
            if too_big:
                continue

            cat = classify_gait(m)
            if not cat:
                continue

            sig = signature(actions)
            key = (cat, round(m["dx"], 2), round(m["lf_move"], 2), round(m["rf_move"], 2), sig)
            if key in seen_local:
                continue
            seen_local.add(key)

            found.append({
                "id": next_id,
                "name": f"v20_{cat}_{next_id:05d}",
                "category": cat,
                "source": path.name,
                "start_frame": start_no,
                "end_frame": end_no,
                "length": len(actions),
                "score": score_skill(cat, m, actions),
                "dx": round(m["dx"], 5),
                "dy": round(m["dy"], 5),
                "left_foot_move": round(m["lf_move"], 5),
                "right_foot_move": round(m["rf_move"], 5),
                "head_z_min": round(m["head_z_min"], 5),
                "head_z_range": round(m["head_z_range"], 5),
                "signature": sig,
                "actions": actions,
            })
            next_id += 1
    return found, next_id


def main() -> None:
    paths = sorted(PARKOUR_JSON.glob("*.json"))
    print("Parkour JSON:", PARKOUR_JSON)
    print("Replays:", len(paths))

    all_skills: list[dict[str, Any]] = []
    next_id = 0
    for idx, path in enumerate(paths, 1):
        skills, next_id = extract_from_replay(path, next_id)
        all_skills.extend(skills)
        if idx % 25 == 0 or idx == len(paths):
            print(f"  {idx}/{len(paths)} replays | raw gait skills={len(all_skills)}")

    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in all_skills:
        by_cat[s["category"]].append(s)

    selected: list[dict[str, Any]] = []
    for cat, arr in by_cat.items():
        # diversité: pas 50 variantes identiques du même replay/signature.
        arr.sort(key=lambda s: s["score"], reverse=True)
        seen = set()
        kept = []
        for s in arr:
            key = (s["source"], s["signature"], round(float(s["dx"]), 1))
            if key in seen:
                continue
            seen.add(key)
            kept.append(s)
            if len(kept) >= MAX_KEEP_PER_CATEGORY:
                break
        selected.extend(kept)

    selected.sort(key=lambda s: (s["category"], -float(s["score"])))
    # remap ids propres
    for i, s in enumerate(selected):
        s["id"] = i

    summary = {
        "version": "20",
        "raw_count": len(all_skills),
        "skill_count": len(selected),
        "categories": Counter(s["category"] for s in selected),
        "source_replays": len(paths),
        "notes": "Gait extraction by foot planted/swing signature, not by raw forward distance.",
    }

    data = {
        "name": "parkour_walking_gait_skills_v20",
        "version": "20",
        "left_foot_idx": LEFT_FOOT_IDX,
        "right_foot_idx": RIGHT_FOOT_IDX,
        "skills": selected,
        "summary": summary,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SKILLS.write_text(json.dumps(data, indent=2), encoding="utf-8")
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, default=lambda x: dict(x)), encoding="utf-8")

    print("\nSaved:", OUT_SKILLS)
    print("Summary:", OUT_SUMMARY)
    print("Categories:", dict(summary["categories"]))
    print("\nTop skills:")
    for cat in ["stand", "lean_forward", "left_step", "right_step", "recover_upright"]:
        arr = [s for s in selected if s["category"] == cat]
        print(" ", cat, len(arr))
        for s in arr[:5]:
            print(
                f"    id={s['id']:4d} score={s['score']:8.2f} dx={s['dx']:6.2f} "
                f"lf={s['left_foot_move']:5.2f} rf={s['right_foot_move']:5.2f} "
                f"head={s['head_z_min']:5.2f} frames={s['start_frame']}-{s['end_frame']} sig={s['signature']}"
            )


if __name__ == "__main__":
    main()
