#!/usr/bin/env python3
"""
extract_locomotion_skills_v18.py

V18 = extraction locomotion-only depuis les vrais replays parkour.

Pourquoi:
  V16.1 a extrait beaucoup de skills humains, mais les meilleurs dx étaient souvent
  des tricks / chutes / glissades. V18 cherche plutôt des segments plus sobres:
    - déplacement vers l'avant modéré
    - tête/torse pas trop bas
    - variation verticale raisonnable
    - peu de spam d'articulations

Entrée:
  datasets/parkour_json/*.json

Sortie:
  generated_replays/parkour_locomotion_skills_v18.json
  generated_replays/parkour_locomotion_skills_v18_summary.json
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
PARKOUR_JSON_DIR = ROOT / "datasets" / "parkour_json"
OUT_DIR = ROOT / "generated_replays"
OUT_SKILLS = OUT_DIR / "parkour_locomotion_skills_v18.json"
OUT_SUMMARY = OUT_DIR / "parkour_locomotion_skills_v18_summary.json"

PLAYER = "0"
WINDOW_SIZES = [3, 4, 5, 6, 8]
MAX_PER_CATEGORY = {
    "stand": 120,
    "lean_forward": 120,
    "step_left": 120,
    "step_right": 120,
    "walk_step": 160,
    "recover_upright": 120,
    "bad_trick_rejected": 80,
}

# Indices POS approximatifs: dans nos JSON Toribash, 0 est souvent head.
HEAD_IDX = 0
CHEST_IDX = 3
L_FOOT_IDX = 19
R_FOOT_IDX = 20

CORE_JOINTS = {0, 1, 2, 3}
ARM_JOINTS = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
LEG_JOINTS = {14, 15, 16, 17, 18, 19}


def as_float3(v: Any) -> tuple[float, float, float] | None:
    try:
        return (float(v[0]), float(v[1]), float(v[2]))
    except Exception:
        return None


def get_pos(player: dict[str, Any], idx: int) -> tuple[float, float, float] | None:
    pos = player.get("pos") or []
    if idx < 0 or idx >= len(pos):
        return None
    return as_float3(pos[idx])


def dist_xy(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.hypot(b[0] - a[0], b[1] - a[1])


def sorted_frames(data: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    frames = data.get("frames", {})
    out = []
    if isinstance(frames, dict):
        for k, f in frames.items():
            try:
                out.append((int(k), f))
            except Exception:
                continue
    elif isinstance(frames, list):
        for i, f in enumerate(frames):
            if isinstance(f, dict):
                out.append((int(f.get("frame", i)), f))
    out.sort(key=lambda x: x[0])
    return out


def player0(frame: dict[str, Any]) -> dict[str, Any] | None:
    players = frame.get("players", {})
    p = players.get(PLAYER) or players.get(0)
    return p if isinstance(p, dict) else None


def joint_pairs(frame: dict[str, Any]) -> list[list[int]]:
    p = player0(frame)
    if not p:
        return []
    pairs = p.get("joint_pairs")
    if isinstance(pairs, list):
        clean = []
        for pair in pairs:
            try:
                j = int(pair[0])
                v = int(pair[1])
                if 0 <= j <= 19 and 1 <= v <= 4:
                    clean.append([j, v])
            except Exception:
                pass
        return clean
    joints = p.get("joints")
    if isinstance(joints, dict):
        clean = []
        for k, v in joints.items():
            try:
                j = int(k)
                val = int(v)
                if 0 <= j <= 19 and 1 <= val <= 4:
                    clean.append([j, val])
            except Exception:
                pass
        return clean
    return []


def signature(actions: list[dict[str, Any]]) -> str:
    c = Counter()
    for a in actions:
        for j, _v in a.get("pairs", []):
            c[int(j)] += 1
    return "-".join(str(j) for j, _ in c.most_common(8))


def action_stats(actions: list[dict[str, Any]]) -> dict[str, float]:
    total = 0
    core = arm = leg = 0
    maxpf = 0
    for a in actions:
        pairs = a.get("pairs", [])
        maxpf = max(maxpf, len(pairs))
        for j, _ in pairs:
            total += 1
            if j in CORE_JOINTS:
                core += 1
            elif j in ARM_JOINTS:
                arm += 1
            elif j in LEG_JOINTS:
                leg += 1
    return {"total": total, "core": core, "arm": arm, "leg": leg, "maxpf": maxpf}


def classify(metrics: dict[str, float], actions: list[dict[str, Any]]) -> tuple[str | None, float, str]:
    dx = metrics["dx"]
    dy = metrics["dy"]
    head0 = metrics["head0"]
    head1 = metrics["head1"]
    head_min = metrics["head_min"]
    head_range = metrics["head_range"]
    chest_range = metrics["chest_range"]
    lateral = abs(dy)
    st = action_stats(actions)

    # rejet trick/chute: grosse variation verticale ou tête très basse sur dataset source
    explosive = st["maxpf"] >= 11 or st["total"] > len(actions) * 8
    too_vertical = head_range > 18 or chest_range > 14
    too_lateral = lateral > max(3.0, abs(dx) * 1.1)

    if dx < -2.0 and not too_vertical:
        score = abs(dx) * 12 + max(0, head_min) - st["maxpf"] * 2
        return "bad_trick_rejected", score, "backward"

    if explosive or too_vertical or too_lateral:
        return None, 0.0, "reject_trick_like"

    # stand: peu de déplacement, hauteur stable, peu d'actions
    if abs(dx) < 0.35 and lateral < 0.35 and head_range < 3.5 and st["total"] <= len(actions) * 5:
        score = 90 - abs(dx) * 30 - lateral * 20 - head_range * 5 - st["maxpf"] * 1.5
        return "stand", score, "stable"

    # recover: la tête remonte ou reste haute après perturbation
    if (head1 - head0) > 1.4 and dx > -0.8 and lateral < 2.8:
        score = (head1 - head0) * 18 + dx * 5 - lateral * 4 - st["maxpf"] * 1.2
        return "recover_upright", score, "head_up"

    # lean forward: petit transfert avant sans chute
    if 0.35 <= dx <= 2.0 and lateral < 1.6 and head_range < 7.0 and st["maxpf"] <= 8:
        score = 80 + dx * 18 - lateral * 8 - head_range * 3 + st["leg"] * 0.5 + st["core"] * 0.8 - st["arm"] * 0.25
        return "lean_forward", score, "controlled_forward"

    # marche: déplacement modéré, pas trop vertical, action jambes dominante mais pas spam total
    if 1.0 <= dx <= 5.5 and lateral < 3.2 and head_range < 10.0 and st["maxpf"] <= 9:
        if st["leg"] >= max(2, st["arm"] * 0.45):
            # pseudo alternance gauche/droite basée sur présence joints 14/16/18 vs 15/17/19
            left = 0
            right = 0
            for a in actions:
                js = {j for j, _ in a.get("pairs", [])}
                if js & {14, 16, 18}:
                    left += 1
                if js & {15, 17, 19}:
                    right += 1
            balance = 1.0 - min(1.0, abs(left - right) / max(1, left + right))
            score = 110 + dx * 20 + balance * 25 - lateral * 6 - head_range * 4 - max(0, st["maxpf"] - 6) * 5
            if left > right + 1:
                return "step_left", score, "left_dominant"
            if right > left + 1:
                return "step_right", score, "right_dominant"
            return "walk_step", score, "balanced_step"

    return None, 0.0, "reject_not_locomotion"


def extract_from_replay(path: Path, next_id: int) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    frames = sorted_frames(data)
    if len(frames) < 4:
        return []

    skills = []
    metadata = data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {}
    mod = metadata.get("mod") or metadata.get("newgame_mod") or "unknown"

    for win in WINDOW_SIZES:
        for start in range(0, len(frames) - win, 1):
            chunk = frames[start : start + win]
            first_no, first = chunk[0]
            last_no, last = chunk[-1]
            p_first = player0(first)
            p_last = player0(last)
            if not p_first or not p_last:
                continue
            h0 = get_pos(p_first, HEAD_IDX)
            h1 = get_pos(p_last, HEAD_IDX)
            c0 = get_pos(p_first, CHEST_IDX)
            c1 = get_pos(p_last, CHEST_IDX)
            if not h0 or not h1 or not c0 or not c1:
                continue

            head_zs = []
            chest_zs = []
            for _, fr in chunk:
                p = player0(fr)
                if not p:
                    continue
                hp = get_pos(p, HEAD_IDX)
                cp = get_pos(p, CHEST_IDX)
                if hp:
                    head_zs.append(hp[2])
                if cp:
                    chest_zs.append(cp[2])
            if not head_zs or not chest_zs:
                continue

            actions = []
            for fno, fr in chunk:
                pairs = joint_pairs(fr)
                if pairs:
                    actions.append({"dt": int(fno - first_no), "pairs": pairs})
            if len(actions) < 2:
                continue

            metrics = {
                "dx": h1[0] - h0[0],
                "dy": h1[1] - h0[1],
                "head0": h0[2],
                "head1": h1[2],
                "head_min": min(head_zs),
                "head_max": max(head_zs),
                "head_range": max(head_zs) - min(head_zs),
                "chest_range": max(chest_zs) - min(chest_zs),
                "distance_xy": dist_xy(h0, h1),
            }
            cat, score, reason = classify(metrics, actions)
            if not cat:
                continue
            if score <= 0:
                continue
            st = action_stats(actions)
            skills.append({
                "id": next_id + len(skills),
                "name": f"{cat}_{next_id + len(skills):06d}",
                "category": cat,
                "source": "real_replay_v18",
                "replay": str(path),
                "mod": mod,
                "start_frame": int(first_no),
                "end_frame": int(last_no),
                "length": len(actions),
                "score": round(float(score), 4),
                "reason": reason,
                "dx": round(metrics["dx"], 4),
                "dy": round(metrics["dy"], 4),
                "head0": round(metrics["head0"], 4),
                "head1": round(metrics["head1"], 4),
                "head_min": round(metrics["head_min"], 4),
                "head_range": round(metrics["head_range"], 4),
                "max_pairs_per_frame": int(st["maxpf"]),
                "total_pairs": int(st["total"]),
                "leg_pairs": int(st["leg"]),
                "arm_pairs": int(st["arm"]),
                "core_pairs": int(st["core"]),
                "signature": signature(actions),
                "actions": actions,
            })
    return skills


def dedupe_and_limit(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_cat = defaultdict(list)
    for s in raw:
        by_cat[s["category"]].append(s)

    final = []
    next_id = 0
    for cat, items in by_cat.items():
        items.sort(key=lambda s: s["score"], reverse=True)
        seen = set()
        kept = []
        for s in items:
            # évite 100 fois le même segment / même signature
            key = (Path(s["replay"]).name, s["start_frame"] // 10, s["end_frame"] // 10, s["signature"])
            loose = (s["signature"], round(s["dx"], 1), round(s["head_range"], 1))
            if key in seen or loose in seen:
                continue
            seen.add(key)
            seen.add(loose)
            ns = dict(s)
            ns["id"] = next_id
            ns["name"] = f"{cat}_{next_id:04d}"
            kept.append(ns)
            next_id += 1
            if len(kept) >= MAX_PER_CATEGORY.get(cat, 100):
                break
        final.extend(kept)
    final.sort(key=lambda s: (s["category"], -s["score"]))
    # renumérote proprement après tri final
    for i, s in enumerate(final):
        s["id"] = i
        s["name"] = f"{s['category']}_{i:04d}"
    return final


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(PARKOUR_JSON_DIR.glob("*.json"))
    print("Parkour JSON:", PARKOUR_JSON_DIR)
    print("Replays:", len(paths))

    raw: list[dict[str, Any]] = []
    for idx, path in enumerate(paths, start=1):
        raw.extend(extract_from_replay(path, len(raw)))
        if idx % 25 == 0 or idx == len(paths):
            print(f"  {idx}/{len(paths)} replays | raw locomotion skills={len(raw)}")

    skills = dedupe_and_limit(raw)
    counts = Counter(s["category"] for s in skills)

    data = {
        "name": "parkour_locomotion_skills_v18",
        "version": 18,
        "source": "real parkour replays locomotion filtered",
        "skill_count": len(skills),
        "categories": dict(counts),
        "window_sizes": WINDOW_SIZES,
        "skills": skills,
    }
    OUT_SKILLS.write_text(json.dumps(data, indent=2), encoding="utf-8")

    summary = {
        "raw_count": len(raw),
        "skill_count": len(skills),
        "categories": dict(counts),
        "top_by_category": {},
    }
    for cat in sorted(counts):
        top = [s for s in skills if s["category"] == cat][:10]
        summary["top_by_category"][cat] = [
            {
                "id": s["id"], "score": s["score"], "dx": s["dx"], "dy": s["dy"],
                "head_min": s["head_min"], "head_range": s["head_range"],
                "frames": [s["start_frame"], s["end_frame"]], "sig": s["signature"],
            }
            for s in top
        ]
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:", OUT_SKILLS)
    print("Summary:", OUT_SUMMARY)
    print("Categories:", dict(counts))
    print("\nTop locomotion skills:")
    for cat in sorted(counts):
        print(" ", cat)
        for s in [x for x in skills if x["category"] == cat][:5]:
            print(
                f"    id={s['id']:4d} score={s['score']:8.2f} dx={s['dx']:6.2f} "
                f"dy={s['dy']:6.2f} head_min={s['head_min']:5.2f} hr={s['head_range']:5.2f} "
                f"pairs={s['total_pairs']:3d} maxpf={s['max_pairs_per_frame']:2d} "
                f"frames={s['start_frame']}-{s['end_frame']} sig={s['signature']}"
            )


if __name__ == "__main__":
    main()
