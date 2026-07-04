#!/usr/bin/env python3
"""
build_xioi_step_signature_v19.py

V19: construit une signature de marche depuis la branche walk_xioi.

But:
  - lire walk_xioi_seed/champion si disponible
  - extraire les patterns d'articulations récurrents
  - écrire une signature utilisée ensuite pour trouver les pas similaires
    dans le dataset parkour.

Sortie:
  generated_replays/walk_xioi_step_signature_v19.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
EVOLUTION = ROOT / "evolution"
OUT_DIR = ROOT / "generated_replays"
OUT_PATH = OUT_DIR / "walk_xioi_step_signature_v19.json"

# Candidats connus de la branche walk_xioi.
SOURCE_CANDIDATES = [
    EVOLUTION / "champion_xioi_mechanic_v7.json",
    EVOLUTION / "walk_xioi_champion_v1.json",
    EVOLUTION / "walk_xioi_seed_v1.json",
    EVOLUTION / "xioi_walk_seed_v1.json",
    OUT_DIR / "walk_xioi_seed_v1.json",
    ROOT / "walk_xioi_seed_v1.json",
]

# Joints utiles pour la locomotion Toribash.
CORE = {0, 1, 2, 3}
ARMS = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13}
LEGS = {14, 15, 16, 17, 18, 19}
IMPORTANT = CORE | LEGS | {4, 5, 6, 7, 8, 9}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def find_source() -> Path:
    for p in SOURCE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Aucune source Xioi trouvée. Cherché:\n" +
        "\n".join(str(p) for p in SOURCE_CANDIDATES)
    )


def extract_commands(data: Any) -> list[dict[str, Any]]:
    """Accepte plusieurs formats déjà rencontrés dans le projet."""
    if isinstance(data, dict):
        for key in ("commands", "actions", "frames"):
            value = data.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                out = []
                for fk, fv in value.items():
                    if isinstance(fv, dict):
                        pairs = fv.get("pairs") or fv.get("joint_pairs")
                        if pairs:
                            out.append({"frame": int(fk), "pairs": pairs})
                return sorted(out, key=lambda x: int(x.get("frame", 0)))
        if "genome" in data and isinstance(data["genome"], list):
            return data["genome"]
    if isinstance(data, list):
        return data
    return []


def normalize_pair(pair: Any) -> tuple[int, int] | None:
    if isinstance(pair, (list, tuple)) and len(pair) >= 2:
        try:
            j = int(pair[0])
            v = int(pair[1])
            if 0 <= j <= 19 and 1 <= v <= 4:
                return j, v
        except Exception:
            return None
    return None


def command_pairs(cmd: dict[str, Any]) -> list[tuple[int, int]]:
    raw = cmd.get("pairs") or cmd.get("joint_pairs") or []
    out = []
    for p in raw:
        q = normalize_pair(p)
        if q:
            out.append(q)
    return out


def signature_for_window(commands: list[dict[str, Any]]) -> dict[str, Any]:
    joint_counts: Counter[int] = Counter()
    value_counts: dict[int, Counter[int]] = defaultdict(Counter)
    pair_counts: Counter[tuple[int, int]] = Counter()
    active_per_turn = []

    for cmd in commands:
        pairs = command_pairs(cmd)
        active_per_turn.append(len(pairs))
        for j, v in pairs:
            joint_counts[j] += 1
            value_counts[j][v] += 1
            pair_counts[(j, v)] += 1

    important_counts = {str(j): joint_counts[j] for j in sorted(IMPORTANT) if joint_counts[j]}
    preferred_values = {
        str(j): value_counts[j].most_common(1)[0][0]
        for j in sorted(value_counts)
        if j in IMPORTANT and value_counts[j]
    }

    return {
        "turn_count": len(commands),
        "avg_active": round(sum(active_per_turn) / max(1, len(active_per_turn)), 3),
        "joint_counts": important_counts,
        "preferred_values": preferred_values,
        "top_pairs": [[j, v, c] for (j, v), c in pair_counts.most_common(40) if j in IMPORTANT],
        "top_joints": [[j, c] for j, c in joint_counts.most_common(20)],
    }


def main() -> None:
    source = find_source()
    data = load_json(source)
    commands = extract_commands(data)
    commands = [c for c in commands if isinstance(c, dict) and command_pairs(c)]

    if not commands:
        raise RuntimeError(f"Aucune commande exploitable dans {source}")

    # Signature globale + sous-fenêtres pour capter les phases de pas.
    global_sig = signature_for_window(commands)

    windows = []
    win = 8
    step = 4
    for i in range(0, max(0, len(commands) - win + 1), step):
        chunk = commands[i:i + win]
        sig = signature_for_window(chunk)
        leg = sum(int(sig["joint_counts"].get(str(j), 0)) for j in LEGS)
        core = sum(int(sig["joint_counts"].get(str(j), 0)) for j in CORE)
        arms = sum(int(sig["joint_counts"].get(str(j), 0)) for j in ARMS)
        score = leg * 2.0 + core * 1.0 + min(arms, leg) * 0.25 - abs(sig["avg_active"] - 5.0) * 2.0
        windows.append({
            "start_index": i,
            "end_index": i + win,
            "score": round(score, 3),
            "signature": sig,
            "actions": [
                {"dt": n, "pairs": command_pairs(cmd)}
                for n, cmd in enumerate(chunk)
            ],
        })

    windows.sort(key=lambda x: x["score"], reverse=True)

    out = {
        "name": "walk_xioi_step_signature_v19",
        "version": 19,
        "source": str(source),
        "command_count": len(commands),
        "important_joints": sorted(IMPORTANT),
        "global_signature": global_sig,
        "prototype_windows": windows[:24],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Source:", source)
    print("Commands:", len(commands))
    print("Saved:", OUT_PATH)
    print("Top joints:", global_sig["top_joints"][:12])
    print("Top pairs:", global_sig["top_pairs"][:12])
    print("Prototype windows:", len(out["prototype_windows"]))
    for w in out["prototype_windows"][:5]:
        print("  window", w["start_index"], "score", w["score"], "avg_active", w["signature"]["avg_active"])


if __name__ == "__main__":
    main()
