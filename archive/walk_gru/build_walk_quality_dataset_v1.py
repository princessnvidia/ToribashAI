#!/usr/bin/env python3
import json
from pathlib import Path
from collections import Counter

ROOT = Path.home() / "Documents" / "ToribashAI"

INPUT_CANDIDATES = [
    ROOT / "datasets/ml/walk_fullbody_sequences_len8.jsonl",
    ROOT / "datasets/ml/parkour_sequences_len8.jsonl",
]

OUT_PATH = ROOT / "datasets/ml/walk_fullbody_quality_v1.jsonl"
SUMMARY_PATH = ROOT / "datasets/ml/walk_fullbody_quality_v1_summary.json"

# Modèle fullbody actuel :
# [4,5,6,7,14,15,16,17,18,19]
# jambes locales = 0..7, bras locaux = 8..9
LEG_IDXS = list(range(8))
ARM_IDXS = [8, 9]

MIN_QUALITY = 0.42


def find_input():
    for p in INPUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Aucun dataset sequence trouvé.")


def get_action(example):
    for key in ["target_action", "action", "y"]:
        if key in example and isinstance(example[key], list):
            return example[key]

    if "actions" in example and isinstance(example["actions"], list) and example["actions"]:
        last = example["actions"][-1]
        if isinstance(last, list):
            return last

    return None


def quality_score(action):
    if not action or len(action) < 10:
        return 0.0, {}

    leg_vals = [action[i] for i in LEG_IDXS]
    arm_vals = [action[i] for i in ARM_IDXS]

    leg_active = sum(1 for v in leg_vals if v != 0)
    arm_active = sum(1 for v in arm_vals if v != 0)
    total_active = sum(1 for v in action if v != 0)

    left_leg = leg_vals[:4]
    right_leg = leg_vals[4:8]

    left_active = sum(1 for v in left_leg if v != 0)
    right_active = sum(1 for v in right_leg if v != 0)

    values = Counter(action)
    dominant_count = values.most_common(1)[0][1]

    score = 0.0

    # On veut des jambes actives.
    score += min(leg_active / 5.0, 1.0) * 0.45

    # On veut éviter les poses mortes.
    score += min(total_active / 7.0, 1.0) * 0.20

    # On veut un minimum gauche/droite.
    if left_active > 0 and right_active > 0:
        score += 0.20

    # Bras utiles mais pas centraux.
    score += min(arm_active / 2.0, 1.0) * 0.05

    # Pénalité si tout est pareil / trop figé.
    if dominant_count >= len(action) - 1:
        score -= 0.25

    # Pénalité si jambes quasi mortes.
    if leg_active <= 1:
        score -= 0.30

    score = max(0.0, min(1.0, score))

    details = {
        "leg_active": leg_active,
        "arm_active": arm_active,
        "total_active": total_active,
        "left_active": left_active,
        "right_active": right_active,
        "dominant_value": values.most_common(1)[0][0],
        "dominant_count": dominant_count,
    }

    return score, details


def main():
    in_path = find_input()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0
    rejected = 0
    missing_action = 0

    quality_bins = Counter()
    active_counter = Counter()

    with in_path.open("r", encoding="utf-8") as f_in, OUT_PATH.open("w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.strip()
            if not line:
                continue

            total += 1
            ex = json.loads(line)

            action = get_action(ex)
            if action is None:
                missing_action += 1
                rejected += 1
                continue

            score, details = quality_score(action)

            bin_key = f"{int(score * 10) / 10:.1f}"
            quality_bins[bin_key] += 1
            active_counter[details.get("leg_active", 0)] += 1

            if score < MIN_QUALITY:
                rejected += 1
                continue

            ex["walk_quality_score"] = round(score, 4)
            ex["walk_quality_details"] = details

            f_out.write(json.dumps(ex, ensure_ascii=False) + "\n")
            kept += 1

    summary = {
        "input": str(in_path),
        "output": str(OUT_PATH),
        "total": total,
        "kept": kept,
        "rejected": rejected,
        "missing_action": missing_action,
        "min_quality": MIN_QUALITY,
        "quality_bins": dict(sorted(quality_bins.items())),
        "leg_active_distribution": dict(sorted(active_counter.items())),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
