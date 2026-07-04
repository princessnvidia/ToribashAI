#!/usr/bin/env python3
"""
export_curated_walking_source_context_rpl_v23_2.py

V23.2: exporte des débuts de marche en conservant le contexte du replay source.

Pourquoi:
  Les exports classic/actions-only donnaient une physique différente (effet lune / pas de vrai contact).
  Ici on garde NEWGAME / ENGAGE / metadata du replay original quand possible, puis on écrit
  uniquement une fenêtre d'actions JOINT au début choisi.

Entrées:
  datasets/parkour_json/*.json

Sorties:
  generated_replays/curated_walk_source_context_v23_2_*.rpl
  copie automatique dans le dossier replay Toribash.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

ROOT = Path.home() / "Documents" / "ToribashAI"
PARKOUR_JSON = ROOT / "datasets" / "parkour_json"
OUT_DIR = ROOT / "generated_replays"
TORIBASH_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)

# Candidats trouvés dans la recherche + curated positifs.
KEYWORDS = [
    "xioi",
    "swex",
    "divine",
    "flarkour",
    "karbn",
    "raid",
    "clay",
    "pigeon",
    "kurr",
    "treasure",
    "pakourxioi",
]

MAX_REPLAYS = 24
START_FRAME_LIMIT = 160
WINDOW_ACTION_FRAMES = 90
MAX_MATCHFRAMES = 900


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def safe_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:90] or "replay"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sorted_frame_items(frames: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    out: list[tuple[int, dict[str, Any]]] = []
    for k, v in frames.items():
        try:
            out.append((int(k), v))
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out


def get_p0(frame: dict[str, Any]) -> dict[str, Any]:
    return frame.get("players", {}).get("0", {}) or frame.get("players", {}).get(0, {}) or {}


def joint_pairs(frame: dict[str, Any]) -> list[list[int]]:
    p0 = get_p0(frame)
    pairs = p0.get("joint_pairs") or []
    cleaned: list[list[int]] = []
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        try:
            j = int(pair[0])
            v = int(pair[1])
        except Exception:
            continue
        if 0 <= j <= 19 and 1 <= v <= 4:
            cleaned.append([j, v])
    return cleaned


def frame_pos(frame: dict[str, Any], idx: int) -> list[float] | None:
    p0 = get_p0(frame)
    pos = p0.get("pos") or []
    if idx < 0 or idx >= len(pos):
        return None
    try:
        return [float(pos[idx][0]), float(pos[idx][1]), float(pos[idx][2])]
    except Exception:
        return None


def center_xy(frame: dict[str, Any]) -> tuple[float, float] | None:
    # centre approximatif: moyenne des premiers bodyparts dispo
    p0 = get_p0(frame)
    pos = p0.get("pos") or []
    pts = []
    for p in pos[:21]:
        try:
            pts.append((float(p[0]), float(p[1])))
        except Exception:
            pass
    if not pts:
        return None
    return (sum(x for x, _ in pts) / len(pts), sum(y for _, y in pts) / len(pts))


def head_z(frame: dict[str, Any]) -> float:
    # idx 0 est souvent head dans nos parsings; fallback max z.
    h = frame_pos(frame, 0)
    if h:
        return h[2]
    p0 = get_p0(frame)
    vals = []
    for p in p0.get("pos") or []:
        try:
            vals.append(float(p[2]))
        except Exception:
            pass
    return max(vals) if vals else 0.0


def score_start_window(items: list[tuple[int, dict[str, Any]]], start_idx: int, end_idx: int) -> float:
    if end_idx <= start_idx:
        return -1e9
    f0 = items[start_idx][1]
    f1 = items[end_idx][1]
    c0 = center_xy(f0)
    c1 = center_xy(f1)
    if not c0 or not c1:
        return -1e9
    dx = c1[0] - c0[0]
    dy = c1[1] - c0[1]
    forward = max(abs(dx), abs(dy))
    hz = sum(head_z(f) for _, f in items[start_idx:end_idx+1]) / (end_idx - start_idx + 1)
    action_frames = sum(1 for _, f in items[start_idx:end_idx+1] if joint_pairs(f))
    # on veut mouvement modéré, tête pas au sol, actions présentes
    too_explosive = max(0.0, forward - 18.0) * 2.0
    too_static = max(0.0, 0.8 - forward) * 8.0
    return forward * 5.0 + min(hz, 40.0) * 0.8 + action_frames * 1.5 - too_explosive - too_static


def choose_window(items: list[tuple[int, dict[str, Any]]]) -> tuple[int, int]:
    if not items:
        return 0, 0
    candidates = []
    for i, (fr, _) in enumerate(items):
        if fr > START_FRAME_LIMIT:
            break
        # fenêtre jusqu'à fr+WINDOW_ACTION_FRAMES environ
        j = i
        while j + 1 < len(items) and items[j + 1][0] <= fr + WINDOW_ACTION_FRAMES:
            j += 1
        if j > i:
            candidates.append((score_start_window(items, i, j), i, j))
    if not candidates:
        return 0, min(len(items) - 1, 20)
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1], candidates[0][2]


def metadata_line(data: dict[str, Any], key: str) -> str | None:
    meta = data.get("metadata") or {}
    val = meta.get(key)
    if val is None:
        return None
    return str(val)


def source_newgame(data: dict[str, Any]) -> str:
    # La plupart de nos parseurs gardent NEWGAME dans metadata; sinon fallback classic.
    meta = data.get("metadata") or {}
    for k in ("NEWGAME", "newgame", "new_game"):
        val = meta.get(k)
        if val:
            s = str(val).strip()
            if s.startswith("NEWGAME"):
                return s
            return "NEWGAME 0;" + s
    mod = meta.get("mod") or meta.get("MOD") or "classic"
    return f"NEWGAME 0;{MAX_MATCHFRAMES} 5 30 0 0 2 100 0 0 0 0 0 0 0 {mod}"


def source_engage(data: dict[str, Any], player: int) -> str:
    meta = data.get("metadata") or {}
    for k in (f"ENGAGE {player}", f"engage_{player}", f"ENGAGE_{player}"):
        val = meta.get(k)
        if val:
            s = str(val).strip()
            if s.startswith("ENGAGE"):
                return s
            return f"ENGAGE {player}; {s}"

    # fallback depuis frame 0 other ENGAGE si présent
    frames = data.get("frames") or {}
    if isinstance(frames, dict) and frames:
        first = frames.get("0") or frames.get(0) or next(iter(frames.values()))
        p = first.get("players", {}).get(str(player), {}) if isinstance(first, dict) else {}
        eng = (p.get("other") or {}).get("ENGAGE")
        if eng:
            return f"ENGAGE {player}; {eng}"

    # fallback safe
    z = 0.0
    y = -3.0 if player == 0 else 0.0
    return f"ENGAGE {player}; 0.000000 {y:.6f} {z:.6f} 0 0 0"


def write_rpl(path: Path, data: dict[str, Any], source_path: Path, items: list[tuple[int, dict[str, Any]]], start_i: int, end_i: int, title: str) -> None:
    start_frame = items[start_i][0]
    lines: list[str] = []
    lines.append("#!/usr/bin/toribash")
    lines.append("#made with toribash-4.92")
    lines.append("#SCORE 0 0")
    lines.append("VERSION 12")
    lines.append(f"FIGHTNAME 0; {title}")
    lines.append("BOUT 0; ToribashAI")
    lines.append("BOUT 1; Uke")
    lines.append("AUTHOR 0; ToribashAI")
    lines.append(source_engage(data, 0))
    lines.append(source_engage(data, 1))
    lines.append(source_newgame(data))
    lines.append("")

    wrote = 0
    for fr, frame in items[start_i:end_i+1]:
        pairs = joint_pairs(frame)
        if not pairs:
            continue
        rel = max(0, fr - start_frame)
        lines.append(f"FRAME {rel};")
        lines.append(f"# source={source_path.name} source_frame={fr}")
        for j, v in pairs:
            lines.append(f"JOINT 0; {j} {v}")
        lines.append("")
        wrote += 1

    # prolongation finale sans POS/QAT, pour laisser la physique continuer.
    last_rel = max(0, items[end_i][0] - start_frame)
    lines.append(f"FRAME {last_rel + 100};")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    print("made:", path.name, "action_frames:", wrote, "source_window:", f"{items[start_i][0]}-{items[end_i][0]}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TORIBASH_REPLAY_DIR.mkdir(parents=True, exist_ok=True)

    paths = []
    for p in sorted(PARKOUR_JSON.glob("*.json")):
        n = norm(p.name)
        if any(k in n for k in KEYWORDS):
            paths.append(p)
    paths = paths[:MAX_REPLAYS]

    print("Source-context curated exports:", len(paths))
    made = []
    for idx, p in enumerate(paths, 1):
        try:
            data = load_json(p)
            frames = data.get("frames") or {}
            if not isinstance(frames, dict):
                continue
            items = sorted_frame_items(frames)
            if len(items) < 5:
                continue
            a, b = choose_window(items)
            title = f"curated_walk_source_context_v23_2_{idx:02d}_{safe_name(p.stem)}"
            out = OUT_DIR / f"{title}.rpl"
            write_rpl(out, data, p, items, a, b, title)
            shutil.copy2(out, TORIBASH_REPLAY_DIR / out.name)
            made.append(out)
        except Exception as e:
            print("ERROR", p.name, e)

    print()
    print("Generated:", len(made))
    print("Copied to:", TORIBASH_REPLAY_DIR)


if __name__ == "__main__":
    main()
