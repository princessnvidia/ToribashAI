#!/usr/bin/env python3
"""
extract_real_replay_skills_v16_1.py

V16.1 = extraction de skills depuis les vrais replays parkour parsés.

Correction V16:
  - frames est un dict {"0": frame, "10": frame, ...}, pas une liste
  - les actions joueur sont dans frames[frame]["players"]["0"]["joint_pairs"]

Sorties:
  generated_replays/parkour_real_replay_skills_v16_1.json
  generated_replays/parkour_real_replay_skills_v16_1_summary.json

But:
  Extraire des fenêtres d'actions humaines réelles et les classer par effet mécanique
  approximatif via les positions POS:
    - forward_impulse
    - walk_step
    - recover
    - stand
    - acro
    - backward_bad
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
PARKOUR_DIR = ROOT / "datasets" / "parkour_json"
OUT_DIR = ROOT / "generated_replays"
OUT_SKILLS = OUT_DIR / "parkour_real_replay_skills_v16_1.json"
OUT_SUMMARY = OUT_DIR / "parkour_real_replay_skills_v16_1_summary.json"

PLAYER = "0"
MAX_SKILLS = 800
WINDOW_ACTIONS_MIN = 3
WINDOW_ACTIONS_MAX = 8
STEP = 2

# Toribash body indices are not perfectly documented here, but empirically in our parsed POS:
# 0..20 body parts. These indices work well enough for rough mechanical scoring.
HEAD_IDX = 0
CHEST_IDX = 1
LUMBAR_IDX = 2
LEFT_FOOT_IDX = 19
RIGHT_FOOT_IDX = 20

LEG_JOINTS = {14, 15, 16, 17, 18, 19}
CORE_JOINTS = {2, 3}
ARM_JOINTS = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13}


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def sorted_frame_items(frames: Any) -> list[tuple[int, dict[str, Any]]]:
    if isinstance(frames, dict):
        out = []
        for k, v in frames.items():
            try:
                out.append((int(k), v))
            except Exception:
                continue
        return sorted(out, key=lambda x: x[0])
    if isinstance(frames, list):
        out = []
        for i, v in enumerate(frames):
            if isinstance(v, dict):
                n = v.get("frame") or v.get("frame_no") or i
                try:
                    n = int(n)
                except Exception:
                    n = i
                out.append((n, v))
        return sorted(out, key=lambda x: x[0])
    return []


def player0(frame: dict[str, Any]) -> dict[str, Any]:
    players = frame.get("players", {})
    return players.get(PLAYER) or players.get(0) or {}


def get_pairs(frame: dict[str, Any]) -> list[list[int]]:
    p0 = player0(frame)
    pairs = p0.get("joint_pairs")
    if isinstance(pairs, list):
        clean = []
        for item in pairs:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                try:
                    j = int(item[0])
                    v = int(item[1])
                except Exception:
                    continue
                if 0 <= j <= 19 and 1 <= v <= 4:
                    clean.append([j, v])
        return clean

    joints = p0.get("joints")
    if isinstance(joints, dict):
        clean = []
        for k, v in joints.items():
            try:
                j = int(k)
                val = int(v)
            except Exception:
                continue
            if 0 <= j <= 19 and 1 <= val <= 4:
                clean.append([j, val])
        return clean

    return []


def get_pos(frame: dict[str, Any], idx: int) -> list[float] | None:
    p0 = player0(frame)
    pos = p0.get("pos")
    if not isinstance(pos, list) or idx >= len(pos):
        return None
    p = pos[idx]
    if not isinstance(p, list) or len(p) < 3:
        return None
    try:
        return [float(p[0]), float(p[1]), float(p[2])]
    except Exception:
        return None


def center_pos(frame: dict[str, Any]) -> list[float] | None:
    p0 = player0(frame)
    pos = p0.get("pos")
    if not isinstance(pos, list) or not pos:
        return None
    pts = []
    for p in pos:
        if isinstance(p, list) and len(p) >= 3:
            try:
                pts.append([float(p[0]), float(p[1]), float(p[2])])
            except Exception:
                pass
    if not pts:
        return None
    return [sum(p[i] for p in pts) / len(pts) for i in range(3)]


def dist_xy(a: list[float], b: list[float]) -> float:
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def action_signature(actions: list[dict[str, Any]]) -> str:
    c = Counter()
    for a in actions:
        for j, _ in a.get("pairs", []):
            c[j] += 1
    return "-".join(str(j) for j, _ in c.most_common(8))


def active_stats(actions: list[dict[str, Any]]) -> dict[str, float]:
    total = 0
    leg = core = arms = 0
    maxpf = 0
    for a in actions:
        pairs = a.get("pairs", [])
        maxpf = max(maxpf, len(pairs))
        for j, _ in pairs:
            total += 1
            if j in LEG_JOINTS:
                leg += 1
            elif j in CORE_JOINTS:
                core += 1
            elif j in ARM_JOINTS:
                arms += 1
    return {
        "total_pairs": total,
        "leg_pairs": leg,
        "core_pairs": core,
        "arm_pairs": arms,
        "max_pairs_per_action": maxpf,
        "avg_pairs_per_action": total / max(1, len(actions)),
    }


def classify(metrics: dict[str, float], stats: dict[str, float]) -> str:
    dx = metrics["dx"]
    dy = metrics["dy"]
    dz_head = metrics["dz_head"]
    dz_center = metrics["dz_center"]
    head_min = metrics["head_min_z"]
    path = metrics["path_xy"]
    rotation_proxy = metrics["rotation_proxy"]
    avg_pairs = stats["avg_pairs_per_action"]
    leg = stats["leg_pairs"]
    core = stats["core_pairs"]
    arms = stats["arm_pairs"]

    # NOTE: dans beaucoup de mods parkour, l'axe utile peut être X ou Y.
    # On garde dx comme forward principal pour notre goal-flat, mais on conserve dy en métrique.
    if dx < -0.9 and abs(dx) > abs(dy) * 0.65:
        return "backward_bad"
    if dx > 1.0 and head_min > 8.5 and leg >= 3:
        if path > 2.2:
            return "forward_impulse"
        return "walk_step"
    if dz_head > 0.55 and dz_center > 0.25 and head_min > 8.0:
        return "recover"
    if abs(dx) < 0.6 and abs(dy) < 0.8 and abs(dz_head) < 0.8 and head_min > 9.0 and avg_pairs <= 5.5:
        return "stand"
    if rotation_proxy > 2.5 or (path > 2.0 and arms > leg * 0.5 and head_min > 7.0):
        return "acro"
    if dx > 0.35 and leg >= 2:
        return "walk_step"
    return "other"


def score_skill(category: str, metrics: dict[str, float], stats: dict[str, float]) -> float:
    dx = metrics["dx"]
    dy = metrics["dy"]
    head_min = metrics["head_min_z"]
    dz_head = metrics["dz_head"]
    path = metrics["path_xy"]
    avg_pairs = stats["avg_pairs_per_action"]
    maxpf = stats["max_pairs_per_action"]
    leg = stats["leg_pairs"]
    core = stats["core_pairs"]
    arms = stats["arm_pairs"]

    s = 0.0
    if category == "forward_impulse":
        s += dx * 35 + path * 8 + min(head_min, 14) * 2
        s += leg * 1.2 + core * 0.8
        s -= abs(dy) * 6
    elif category == "walk_step":
        s += dx * 28 + min(head_min, 14) * 2.5
        s += leg * 1.0 + core * 0.7
        s -= abs(dy) * 5
    elif category == "recover":
        s += dz_head * 35 + metrics["dz_center"] * 25 + min(head_min, 14) * 2
        s += core * 1.2 + leg * 0.8
    elif category == "stand":
        s += min(head_min, 14) * 5
        s -= abs(dx) * 10 + abs(dy) * 8
        s -= max(0, avg_pairs - 4) * 4
    elif category == "acro":
        s += path * 10 + metrics["rotation_proxy"] * 12 + min(head_min, 14)
    elif category == "backward_bad":
        s += abs(dx) * 10
    else:
        s += dx * 8 + min(head_min, 14)

    s -= max(0, maxpf - 7) * 5
    s -= max(0, arms - leg * 1.5) * 1.5
    return round(float(s), 4)


def extract_from_replay(path: Path, skill_start_id: int) -> list[dict[str, Any]]:
    data = load_json(path)
    if not data:
        return []

    items = sorted_frame_items(data.get("frames"))
    if len(items) < 8:
        return []

    # garde uniquement les frames avec actions, mais conserve la frame complète pour métriques.
    action_indices = []
    for idx, (frame_no, frame) in enumerate(items):
        pairs = get_pairs(frame)
        if pairs:
            action_indices.append(idx)

    skills = []
    seen = set()

    for start_pos in range(0, max(0, len(action_indices) - WINDOW_ACTIONS_MIN), STEP):
        for win in (3, 4, 5, 6, 8):
            if start_pos + win > len(action_indices):
                continue
            selected_idx = action_indices[start_pos : start_pos + win]
            start_i = selected_idx[0]
            end_i = selected_idx[-1]

            start_frame_no, start_frame = items[start_i]
            end_frame_no, end_frame = items[end_i]

            c0 = center_pos(start_frame)
            c1 = center_pos(end_frame)
            h0 = get_pos(start_frame, HEAD_IDX)
            h1 = get_pos(end_frame, HEAD_IDX)
            chest0 = get_pos(start_frame, CHEST_IDX)
            chest1 = get_pos(end_frame, CHEST_IDX)
            lumbar0 = get_pos(start_frame, LUMBAR_IDX)
            lumbar1 = get_pos(end_frame, LUMBAR_IDX)

            if not (c0 and c1 and h0 and h1):
                continue

            head_zs = []
            centers = []
            for ii in range(start_i, end_i + 1):
                hp = get_pos(items[ii][1], HEAD_IDX)
                cp = center_pos(items[ii][1])
                if hp:
                    head_zs.append(hp[2])
                if cp:
                    centers.append(cp)

            if not head_zs or not centers:
                continue

            actions = []
            base_frame = start_frame_no
            for ii in selected_idx:
                frame_no, frame = items[ii]
                actions.append({
                    "dt": int(round((frame_no - base_frame) / 5)),
                    "frame": int(frame_no - base_frame),
                    "pairs": get_pairs(frame),
                })

            sig = action_signature(actions)
            dedupe_key = (sig, win, round(c1[0] - c0[0], 1), round(c1[1] - c0[1], 1))
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            rotation_proxy = 0.0
            if chest0 and chest1 and lumbar0 and lumbar1:
                # Approximation: changement relatif chest-lumbar en XY/Z.
                v0 = [chest0[i] - lumbar0[i] for i in range(3)]
                v1 = [chest1[i] - lumbar1[i] for i in range(3)]
                rotation_proxy = math.sqrt(sum((v1[i] - v0[i]) ** 2 for i in range(3)))

            metrics = {
                "dx": c1[0] - c0[0],
                "dy": c1[1] - c0[1],
                "dz_center": c1[2] - c0[2],
                "dz_head": h1[2] - h0[2],
                "head_start_z": h0[2],
                "head_end_z": h1[2],
                "head_min_z": min(head_zs),
                "head_avg_z": mean(head_zs),
                "center_start_z": c0[2],
                "center_end_z": c1[2],
                "path_xy": sum(dist_xy(centers[i], centers[i + 1]) for i in range(len(centers) - 1)),
                "rotation_proxy": rotation_proxy,
                "duration_frames": end_frame_no - start_frame_no,
            }
            stats = active_stats(actions)
            category = classify(metrics, stats)

            if category == "other":
                continue

            skill = {
                "id": skill_start_id + len(skills),
                "name": f"real_{category}_{skill_start_id + len(skills):04d}",
                "source": "real_replay",
                "replay": str(path),
                "replay_name": path.name,
                "start_frame": int(start_frame_no),
                "end_frame": int(end_frame_no),
                "length": len(actions),
                "category": category,
                "score": score_skill(category, metrics, stats),
                "signature": sig,
                "metrics": metrics,
                "stats": stats,
                "actions": actions,
            }
            skills.append(skill)

    return skills


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PARKOUR_DIR.glob("*.json"))
    print("Parkour JSON:", PARKOUR_DIR)
    print("Replays:", len(files))

    all_skills = []
    errors = 0

    for i, path in enumerate(files, 1):
        try:
            skills = extract_from_replay(path, len(all_skills))
            all_skills.extend(skills)
        except Exception as e:
            errors += 1
            if errors <= 8:
                print("ERROR", path.name, repr(e))

        if i % 25 == 0 or i == len(files):
            print(f"  {i}/{len(files)} replays | raw skills={len(all_skills)}")

    # Tri par catégories utiles d'abord, puis score.
    priority = {
        "stand": 0,
        "recover": 1,
        "forward_impulse": 2,
        "walk_step": 3,
        "acro": 4,
        "backward_bad": 9,
    }
    all_skills.sort(key=lambda s: (priority.get(s["category"], 8), -s["score"]))

    # Garde un mélange par catégorie pour éviter que forward ou bad écrase tout.
    caps = {
        "stand": 140,
        "recover": 160,
        "forward_impulse": 180,
        "walk_step": 200,
        "acro": 80,
        "backward_bad": 40,
    }
    kept = []
    counts = Counter()
    seen_sigs = set()
    for s in all_skills:
        cat = s["category"]
        if counts[cat] >= caps.get(cat, 50):
            continue
        # dédoublonnage léger mais pas trop strict.
        key = (cat, s["signature"], round(s["metrics"]["dx"], 1), round(s["metrics"]["dy"], 1))
        if key in seen_sigs:
            continue
        seen_sigs.add(key)
        ns = dict(s)
        ns["id"] = len(kept)
        ns["name"] = f"real_{cat}_{len(kept):04d}"
        kept.append(ns)
        counts[cat] += 1
        if len(kept) >= MAX_SKILLS:
            break

    out = {
        "name": "parkour_real_replay_skills_v16_1",
        "version": "16.1",
        "source": "real_parkour_parsed_json",
        "turnframes": 5,
        "skill_count": len(kept),
        "categories": dict(Counter(s["category"] for s in kept)),
        "skills": kept,
    }

    summary = {
        "version": "16.1",
        "replays": len(files),
        "errors": errors,
        "raw_skill_count": len(all_skills),
        "kept_skill_count": len(kept),
        "categories": dict(Counter(s["category"] for s in kept)),
        "top_by_category": {},
    }
    for cat in sorted(summary["categories"]):
        summary["top_by_category"][cat] = [
            {
                "id": s["id"],
                "score": s["score"],
                "replay_name": s["replay_name"],
                "start_frame": s["start_frame"],
                "end_frame": s["end_frame"],
                "dx": round(s["metrics"]["dx"], 3),
                "dy": round(s["metrics"]["dy"], 3),
                "head_min_z": round(s["metrics"]["head_min_z"], 3),
                "signature": s["signature"],
            }
            for s in sorted([x for x in kept if x["category"] == cat], key=lambda x: x["score"], reverse=True)[:8]
        ]

    OUT_SKILLS.write_text(json.dumps(out, indent=2), encoding="utf-8")
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:", OUT_SKILLS)
    print("Summary:", OUT_SUMMARY)
    print("Categories:", out["categories"])
    print("\nTop skills:")
    for cat, examples in summary["top_by_category"].items():
        print(" ", cat)
        for ex in examples[:5]:
            print(
                f"    id={ex['id']:4d} score={ex['score']:8.2f} "
                f"dx={ex['dx']:6.2f} dy={ex['dy']:6.2f} head={ex['head_min_z']:5.2f} "
                f"frames={ex['start_frame']}-{ex['end_frame']} sig={ex['signature']}"
            )


if __name__ == "__main__":
    main()
