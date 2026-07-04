#!/usr/bin/env python3
from pathlib import Path
import json
import math
from statistics import mean

PROJECT = Path.home() / "Documents" / "ToribashAI"

IN_DIR = PROJECT / "datasets" / "parkour_json"
OUT_DIR = PROJECT / "datasets" / "motifs"

OUT_JSONL = OUT_DIR / "forward_motifs_v1.jsonl"
OUT_SUMMARY = OUT_DIR / "forward_motifs_v1_summary.json"

WINDOW_SIZE = 12
MIN_DISPLACEMENT_XY = 3.0
MIN_SPEED_XY_PER_FRAME = 0.015
MAX_HEIGHT_LOSS = 8.0
MIN_ACTION_CHANGES = 3


def body_center_from_pos(pos):
    if not pos:
        return None

    points = []

    if isinstance(pos, list):
        if len(pos) > 0 and isinstance(pos[0], list):
            points = pos
        elif len(pos) >= 3:
            points = [pos[i:i + 3] for i in range(0, len(pos) - 2, 3)]

    if not points:
        return None

    xs = [p[0] for p in points if len(p) >= 3]
    ys = [p[1] for p in points if len(p) >= 3]
    zs = [p[2] for p in points if len(p) >= 3]

    if not xs:
        return None

    return [mean(xs), mean(ys), mean(zs)]


def get_frame_number(frame):
    return int(frame.get("frame", frame.get("frame_number", 0)))


def get_pos(frame):
    player = frame.get("players", {}).get("0", {})
    return (
        player.get("pos")
        or player.get("POS")
        or player.get("positions")
        or player.get("body_pos")
    )

def get_actions(frame):
    player = frame.get("players", {}).get("0", {})
    actions = player.get("joints", {})

    arr = [0] * 20
    for k, v in actions.items():
        jid = int(k)
        if 0 <= jid < 20:
            arr[jid] = int(v)

    return arr


def dist_xy(a, b):
    return math.sqrt((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2)


def action_change_count(actions_seq):
    count = 0

    for i in range(1, len(actions_seq)):
        prev = actions_seq[i - 1]
        cur = actions_seq[i]

        for a, b in zip(prev, cur):
            if a != b:
                count += 1

    return count


def load_replay(path):
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    frames_raw = data.get("frames", {})
    if not frames_raw:
        return None

    if isinstance(frames_raw, dict):
        frames = []
        for frame_id, frame_data in frames_raw.items():
            if isinstance(frame_data, dict):
                frame_data = dict(frame_data)
                frame_data["frame"] = int(frame_id)
                frames.append(frame_data)

        frames.sort(key=lambda x: x["frame"])
        return data, frames

    if isinstance(frames_raw, list):
        return data, frames_raw

    return None


def extract_from_replay(path):
    loaded = load_replay(path)
    if loaded is None:
        return []

    data, frames = loaded

    usable = []

    for frame in frames:
        center = body_center_from_pos(get_pos(frame))
        actions = get_actions(frame)

        if center is None or actions is None:
            continue

        usable.append({
            "frame": get_frame_number(frame),
            "center": center,
            "actions": actions,
        })

    motifs = []

    if len(usable) < WINDOW_SIZE + 1:
        return motifs

    for start in range(0, len(usable) - WINDOW_SIZE):
        window = usable[start:start + WINDOW_SIZE]

        first = window[0]
        last = window[-1]

        dx = last["center"][0] - first["center"][0]
        dy = last["center"][1] - first["center"][1]
        dz = last["center"][2] - first["center"][2]

        displacement_xy = dist_xy(first["center"], last["center"])
        dt = max(1, last["frame"] - first["frame"])
        speed_xy = displacement_xy / dt

        actions_seq = [w["actions"] for w in window]
        changes = action_change_count(actions_seq)

        if displacement_xy < MIN_DISPLACEMENT_XY:
            continue

        if speed_xy < MIN_SPEED_XY_PER_FRAME:
            continue

        if dz < -MAX_HEIGHT_LOSS:
            continue

        if changes < MIN_ACTION_CHANGES:
            continue

        motifs.append({
            "source_file": str(path),
            "source_name": path.name,
            "start_frame": first["frame"],
            "end_frame": last["frame"],
            "frames": [w["frame"] for w in window],
            "start_center": first["center"],
            "end_center": last["center"],
            "dx": dx,
            "dy": dy,
            "dz": dz,
            "displacement_xy": displacement_xy,
            "speed_xy_per_frame": speed_xy,
            "action_changes": changes,
            "actions": actions_seq,
            "score": displacement_xy + speed_xy * 100.0 - max(0.0, -dz) * 0.2,
        })

    return motifs


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files = sorted(IN_DIR.glob("*.json"))

    if not files:
        raise FileNotFoundError(f"Aucun JSON trouvé dans {IN_DIR}")

    all_motifs = []
    errors = []

    print(f"Input: {IN_DIR}")
    print(f"Files: {len(files)}")
    print("Extraction des motifs d'avancée...")

    for i, path in enumerate(files, start=1):
        try:
            motifs = extract_from_replay(path)
            all_motifs.extend(motifs)
            print(f"[{i}/{len(files)}] {path.name}: {len(motifs)} motifs")
        except Exception as e:
            errors.append({
                "file": str(path),
                "error": repr(e),
            })
            print(f"[ERREUR] {path.name}: {e}")

    all_motifs.sort(key=lambda x: x["score"], reverse=True)

    with OUT_JSONL.open("w", encoding="utf-8") as f:
        for motif in all_motifs:
            f.write(json.dumps(motif, ensure_ascii=False) + "\n")

    summary = {
        "input_dir": str(IN_DIR),
        "output_jsonl": str(OUT_JSONL),
        "files": len(files),
        "motifs": len(all_motifs),
        "errors": len(errors),
        "window_size": WINDOW_SIZE,
        "min_displacement_xy": MIN_DISPLACEMENT_XY,
        "min_speed_xy_per_frame": MIN_SPEED_XY_PER_FRAME,
        "max_height_loss": MAX_HEIGHT_LOSS,
        "min_action_changes": MIN_ACTION_CHANGES,
        "top_20": all_motifs[:20],
        "error_samples": errors[:20],
    }

    with OUT_SUMMARY.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print("Terminé.")
    print(f"Motifs trouvés: {len(all_motifs)}")
    print(f"JSONL: {OUT_JSONL}")
    print(f"Summary: {OUT_SUMMARY}")

    if all_motifs:
        best = all_motifs[0]
        print()
        print("MEILLEUR MOTIF:")
        print(best["source_name"])
        print("frames:", best["start_frame"], "->", best["end_frame"])
        print("displacement_xy:", round(best["displacement_xy"], 3))
        print("dx/dy/dz:", round(best["dx"], 3), round(best["dy"], 3), round(best["dz"], 3))
        print("score:", round(best["score"], 3))


if __name__ == "__main__":
    main()
