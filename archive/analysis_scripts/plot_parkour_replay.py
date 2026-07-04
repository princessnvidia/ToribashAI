#!/usr/bin/env python3
from pathlib import Path
import json
import sys
import matplotlib.pyplot as plt

BODY_INDEX = 0
PLAYER_ID = "0"


def load_positions(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    frames = data.get("frames", {})
    frame_ids = sorted(int(k) for k in frames.keys())

    xs, ys, zs, fs = [], [], [], []

    for fid in frame_ids:
        frame = frames[str(fid)]
        player = frame.get("players", {}).get(PLAYER_ID)
        if not player:
            continue

        pos = player.get("pos", [])
        if len(pos) <= BODY_INDEX:
            continue

        x, y, z = pos[BODY_INDEX]
        xs.append(x)
        ys.append(y)
        zs.append(z)
        fs.append(fid)

    return data, fs, xs, ys, zs


def plot(path):
    data, frames, xs, ys, zs = load_positions(path)

    if not xs:
        print("Aucune position trouvée.")
        return

    meta = data.get("metadata", {})
    title = meta.get("fightname", Path(path).stem)
    mod = meta.get("mod", "")

    out_dir = Path.home() / "Documents/ToribashAI/plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(path).stem[:120]

    plt.figure()
    plt.plot(xs, ys)
    plt.scatter([xs[0]], [ys[0]], label="départ")
    plt.scatter([xs[-1]], [ys[-1]], label="arrivée")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Trajectoire XY\n{title}\n{mod}")
    plt.legend()
    plt.savefig(out_dir / f"{safe_name}_xy.png", dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(ys, zs)
    plt.scatter([ys[0]], [zs[0]], label="départ")
    plt.scatter([ys[-1]], [zs[-1]], label="arrivée")
    plt.xlabel("Y")
    plt.ylabel("Z hauteur")
    plt.title(f"Profil hauteur Y/Z\n{title}\n{mod}")
    plt.legend()
    plt.savefig(out_dir / f"{safe_name}_yz.png", dpi=160, bbox_inches="tight")
    plt.close()

    print("Replay:", title)
    print("Mod:", mod)
    print("Frames:", len(frames))
    print("Images sauvées dans:", out_dir)


def main():
    if len(sys.argv) < 2:
        print("Usage: plot_parkour_replay.py replay.json")
        sys.exit(1)

    plot(sys.argv[1])


if __name__ == "__main__":
    main()
