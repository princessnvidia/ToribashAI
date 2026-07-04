#!/usr/bin/env python3
"""
extract_xioi_like_steps_v19.py

Cherche dans datasets/parkour_json les fenêtres qui ressemblent à la signature
Xioi walking loop. Produit un dataset de pas uniquement.

Entrée:
  generated_replays/walk_xioi_step_signature_v19.json
  datasets/parkour_json/*.json

Sortie:
  generated_replays/xioi_like_step_skills_v19.json
  generated_replays/xioi_like_step_skills_v19_summary.json
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
PARKOUR_JSON = ROOT / "datasets" / "parkour_json"
OUT_DIR = ROOT / "generated_replays"
SIGNATURE_PATH = OUT_DIR / "walk_xioi_step_signature_v19.json"
OUT_PATH = OUT_DIR / "xioi_like_step_skills_v19.json"
SUMMARY_PATH = OUT_DIR / "xioi_like_step_skills_v19_summary.json"

CORE = {0, 1, 2, 3}
ARMS = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
LEGS = {14, 15, 16, 17, 18, 19}
IMPORTANT = CORE | LEGS | {4, 5, 6, 7, 8, 9}

WINDOW_SIZES = [4, 6, 8, 10]
MAX_PER_CATEGORY = 220
MAX_TOTAL = 900


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sorted_frames(data: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    frames = data.get("frames", {})
    if not isinstance(frames, dict):
        return []
    out = []
    for k, f in frames.items():
        try:
            out.append((int(k), f))
        except Exception:
            pass
    return sorted(out, key=lambda x: x[0])


def player0(frame: dict[str, Any]) -> dict[str, Any]:
    return frame.get("players", {}).get("0", {}) if isinstance(frame, dict) else {}


def pairs(frame: dict[str, Any]) -> list[tuple[int, int]]:
    raw = player0(frame).get("joint_pairs", [])
    out = []
    for p in raw:
        try:
            j, v = int(p[0]), int(p[1])
            if 0 <= j <= 19 and 1 <= v <= 4:
                out.append((j, v))
        except Exception:
            continue
    return out


def pos_center(frame: dict[str, Any]) -> tuple[float, float, float] | None:
    pos = player0(frame).get("pos")
    if not pos or not isinstance(pos, list):
        return None
    pts = []
    for p in pos:
        if isinstance(p, list) and len(p) >= 3:
            try:
                pts.append((float(p[0]), float(p[1]), float(p[2])))
            except Exception:
                pass
    if not pts:
        return None
    return (
        sum(p[0] for p in pts) / len(pts),
        sum(p[1] for p in pts) / len(pts),
        sum(p[2] for p in pts) / len(pts),
    )


def head_z(frame: dict[str, Any]) -> float:
    pos = player0(frame).get("pos")
    if isinstance(pos, list) and pos:
        try:
            return max(float(p[2]) for p in pos if isinstance(p, list) and len(p) >= 3)
        except Exception:
            pass
    c = pos_center(frame)
    return c[2] if c else 0.0


def action_signature(chunk: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    joint_counts: Counter[int] = Counter()
    value_counts: Counter[tuple[int, int]] = Counter()
    active_counts = []
    actions = []
    for idx, (fr, f) in enumerate(chunk):
        ps = pairs(f)
        active_counts.append(len(ps))
        actions.append({"dt": idx, "source_frame": fr, "pairs": [[j, v] for j, v in ps]})
        for j, v in ps:
            if j in IMPORTANT:
                joint_counts[j] += 1
                value_counts[(j, v)] += 1
    return {
        "joint_counts": joint_counts,
        "value_counts": value_counts,
        "avg_active": sum(active_counts) / max(1, len(active_counts)),
        "actions": actions,
    }


def cosine(a: Counter[int], b: Counter[int]) -> float:
    keys = set(a) | set(b)
    dot = sum(a[k] * b[k] for k in keys)
    na = math.sqrt(sum(a[k] ** 2 for k in keys))
    nb = math.sqrt(sum(b[k] ** 2 for k in keys))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def load_target_counter(sig: dict[str, Any]) -> Counter[int]:
    jc = sig["global_signature"].get("joint_counts", {})
    c = Counter()
    for k, v in jc.items():
        try:
            c[int(k)] = float(v)
        except Exception:
            pass
    return c


def classify(dx: float, dy: float, avg_head: float, z_span: float, sim: float, avg_active: float) -> str | None:
    # marche/pas = déplacement modéré, pas une explosion/trick.
    if sim < 0.28:
        return None
    if avg_active < 1.0 or avg_active > 9.5:
        return None
    if z_span > 16.0:
        return None
    if abs(dy) > 10.0:
        return None

    if -0.35 <= dx <= 0.55 and z_span < 4.0:
        return "stand_like"
    if 0.20 <= dx <= 2.8 and avg_head > 5.0:
        return "walk_step_like"
    if 2.8 < dx <= 5.5 and avg_head > 4.5:
        return "forward_shift_like"
    if -1.2 <= dx <= 1.2 and avg_head > 6.0 and z_span < 8.0:
        return "recover_like"
    return None


def main() -> None:
    sig = load_json(SIGNATURE_PATH)
    target = load_target_counter(sig)
    if not target:
        raise RuntimeError("Signature Xioi vide. Lance build_xioi_step_signature_v19.py avant.")

    raw = []
    files = sorted(PARKOUR_JSON.glob("*.json"))
    print("Signature:", SIGNATURE_PATH)
    print("Parkour replays:", len(files))

    for fi, path in enumerate(files, 1):
        try:
            data = load_json(path)
        except Exception:
            continue
        frames = sorted_frames(data)
        if len(frames) < 12:
            continue

        for win in WINDOW_SIZES:
            for start in range(0, len(frames) - win, max(1, win // 2)):
                chunk = frames[start:start + win]
                c0 = pos_center(chunk[0][1])
                c1 = pos_center(chunk[-1][1])
                if not c0 or not c1:
                    continue
                dx = c1[0] - c0[0]
                dy = c1[1] - c0[1]
                heads = [head_z(f) for _, f in chunk]
                avg_head = sum(heads) / len(heads)
                z_span = max(heads) - min(heads)
                s = action_signature(chunk)
                sim = cosine(s["joint_counts"], target)
                cat = classify(dx, dy, avg_head, z_span, sim, s["avg_active"])
                if not cat:
                    continue

                # Score volontairement pas juste dx: on veut du pas stable.
                score = (
                    sim * 160.0
                    + max(0.0, min(dx, 3.2)) * 18.0
                    + min(avg_head, 20.0) * 1.2
                    - z_span * 4.0
                    - abs(dy) * 3.0
                    - abs(s["avg_active"] - 4.5) * 4.0
                )

                raw.append({
                    "category": cat,
                    "score": round(score, 3),
                    "similarity": round(sim, 4),
                    "dx": round(dx, 4),
                    "dy": round(dy, 4),
                    "avg_head": round(avg_head, 4),
                    "z_span": round(z_span, 4),
                    "avg_active": round(s["avg_active"], 3),
                    "source_replay": str(path),
                    "start_frame": chunk[0][0],
                    "end_frame": chunk[-1][0],
                    "length": win,
                    "signature": "-".join(str(j) for j, _ in s["joint_counts"].most_common(10)),
                    "actions": s["actions"],
                })

        if fi % 25 == 0 or fi == len(files):
            print(f"  {fi}/{len(files)} | raw={len(raw)}")

    # Diversité: limite les signatures/replays dominants.
    raw.sort(key=lambda x: x["score"], reverse=True)
    selected = []
    per_cat = Counter()
    per_sig = Counter()
    per_replay = Counter()
    for item in raw:
        cat = item["category"]
        sigkey = (cat, item["signature"])
        replay = item["source_replay"]
        if per_cat[cat] >= MAX_PER_CATEGORY:
            continue
        if per_sig[sigkey] >= 8:
            continue
        if per_replay[replay] >= 12:
            continue
        item = dict(item)
        item["id"] = len(selected)
        selected.append(item)
        per_cat[cat] += 1
        per_sig[sigkey] += 1
        per_replay[replay] += 1
        if len(selected) >= MAX_TOTAL:
            break

    out = {
        "name": "xioi_like_step_skills_v19",
        "version": 19,
        "signature_source": str(SIGNATURE_PATH),
        "skill_count": len(selected),
        "categories": dict(per_cat),
        "skills": selected,
    }
    summary = {
        "raw_count": len(raw),
        "selected_count": len(selected),
        "categories": dict(per_cat),
        "top": {
            cat: [s for s in selected if s["category"] == cat][:8]
            for cat in sorted(per_cat)
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Saved:", OUT_PATH)
    print("Summary:", SUMMARY_PATH)
    print("Categories:", dict(per_cat))
    print("Top skills:")
    for cat in sorted(per_cat):
        print(" ", cat)
        for s in [x for x in selected if x["category"] == cat][:5]:
            print(f"    id={s['id']:4d} score={s['score']:8.2f} sim={s['similarity']:.3f} dx={s['dx']:6.2f} dy={s['dy']:6.2f} head={s['avg_head']:5.2f} zspan={s['z_span']:5.2f} frames={s['start_frame']}-{s['end_frame']}")


if __name__ == "__main__":
    main()
