#!/usr/bin/env python3
"""
generate_xioi_loop_phase_calibration_v50.py

V50 = réglage fin de phase pour la marche Xioi.

But:
  - partir du replay V49 / V48 / champion qui contient déjà une loop quasi correcte
  - générer plusieurs RPL courts avec un cycle raccourci
  - tester visuellement lequel recale le pied sans retard cumulatif

Hypothèse actuelle:
  loop de base observée: 485 -> 750, mais un poil trop longue.
  On génère donc plusieurs variantes: 485->750, 745, 740, 735, 730, 725.

Sorties:
  generated_replays/xioi_loop_phase_v50_lenXXX.rpl
  copies dans Toribash replay/ et replay/parkour/
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"

TORIBASH_DIR = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
REPLAY_ROOT = TORIBASH_DIR / "replay"
REPLAY_PARKOUR = REPLAY_ROOT / "parkour"

# On préfère utiliser le V49 champion si présent, sinon base, sinon V48/V37.
SOURCE_CANDIDATES = [
    OUT_DIR / "xioi_stable_loop_v49_champion_candidate.rpl",
    OUT_DIR / "xioi_stable_loop_v49_base.rpl",
    OUT_DIR / "xioi_assassin_template_loop_v48.rpl",
    OUT_DIR / "xioi_427_assassincreedhunter_v37.rpl",
]

MOD_NAME = "Urban_Structure/assassincreedhunter.tbm"

# Segment visuel quasi correct, à raccourcir pour compenser le retard.
LOOP_START = 485
LOOP_END_VARIANTS = [750, 745, 740, 735, 730, 725, 720]

# On garde un replay court et lisible dans l'UI Toribash.
PRELUDE_END = 315
LOOP_REPEATS = 5
TURNFRAMES = 5

FRAME_RE = re.compile(r"^FRAME\s+(-?\d+)\s*;")
NEWGAME_RE = re.compile(r"^NEWGAME\s+0;")
FIGHTNAME_RE = re.compile(r"^FIGHTNAME\s+0;")


def pick_source() -> Path:
    for p in SOURCE_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Aucun replay source trouvé. Cherché:\n" + "\n".join(str(p) for p in SOURCE_CANDIDATES)
    )


def split_blocks(lines: List[str]) -> Tuple[List[str], Dict[int, List[str]]]:
    header: List[str] = []
    blocks: Dict[int, List[str]] = {}
    current_frame = None
    current_block: List[str] = []

    def flush():
        nonlocal current_frame, current_block
        if current_frame is not None:
            blocks[current_frame] = current_block
        current_frame = None
        current_block = []

    for line in lines:
        m = FRAME_RE.match(line)
        if m:
            flush()
            current_frame = int(m.group(1))
            current_block = [line]
        else:
            if current_frame is None:
                header.append(line)
            else:
                current_block.append(line)
    flush()
    return header, blocks


def rewrite_header(header: List[str], name: str, matchframes: int) -> List[str]:
    out: List[str] = []
    saw_fight = False
    saw_newgame = False

    for line in header:
        if FIGHTNAME_RE.match(line):
            out.append(f"FIGHTNAME 0; {name}")
            saw_fight = True
        elif NEWGAME_RE.match(line):
            # NEWGAME 0;matchframes turnframes ... mod
            parts = line.split(";")
            rest = parts[1].strip().split() if len(parts) > 1 else []
            if len(rest) >= 2:
                rest[0] = str(matchframes)
                rest[1] = str(TURNFRAMES)
                if rest[-1].endswith(".tbm") or "/" in rest[-1] or rest[-1] == "classic":
                    rest[-1] = MOD_NAME
                else:
                    rest.append(MOD_NAME)
                out.append("NEWGAME 0;" + " ".join(rest))
            else:
                out.append(f"NEWGAME 0;{matchframes} {TURNFRAMES} 30 0 0 2 100 0 0 0 0 0 0 0 {MOD_NAME}")
            saw_newgame = True
        else:
            out.append(line)

    if not saw_fight:
        insert_at = min(5, len(out))
        out.insert(insert_at, f"FIGHTNAME 0; {name}")
    if not saw_newgame:
        out.append(f"NEWGAME 0;{matchframes} {TURNFRAMES} 30 0 0 2 100 0 0 0 0 0 0 0 {MOD_NAME}")

    return out


def retime_block(block: List[str], new_frame: int) -> List[str]:
    out = []
    replaced = False
    for line in block:
        if not replaced and FRAME_RE.match(line):
            out.append(f"FRAME {new_frame}; 0 0 0 0")
            replaced = True
        else:
            out.append(line)
    return out


def nearest_frames(blocks: Dict[int, List[str]], start: int, end: int) -> List[int]:
    return [f for f in sorted(blocks) if start <= f <= end]


def build_variant(source: Path, header: List[str], blocks: Dict[int, List[str]], loop_end: int) -> Path:
    loop_frames = nearest_frames(blocks, LOOP_START, loop_end)
    prelude_frames = nearest_frames(blocks, 0, PRELUDE_END)

    if len(loop_frames) < 5:
        raise RuntimeError(f"Pas assez de frames loop pour {LOOP_START}->{loop_end}: {len(loop_frames)}")
    if len(prelude_frames) < 5:
        raise RuntimeError(f"Pas assez de frames prelude 0->{PRELUDE_END}: {len(prelude_frames)}")

    loop_len = loop_end - LOOP_START
    name = f"xioi_loop_phase_v50_len{loop_len}"

    # Durée courte: préambule exact puis quelques cycles retimés juste après.
    # On garde les offsets internes du segment source, mais on commence après PRELUDE_END + TURNFRAMES.
    output_blocks: List[List[str]] = []

    for f in prelude_frames:
        output_blocks.append(retime_block(blocks[f], f))

    next_start = PRELUDE_END + TURNFRAMES
    for rep in range(LOOP_REPEATS):
        for f in loop_frames:
            rel = f - LOOP_START
            new_f = next_start + rep * (loop_len + TURNFRAMES) + rel
            output_blocks.append(retime_block(blocks[f], new_f))

    last_frame = 0
    for b in output_blocks:
        m = FRAME_RE.match(b[0])
        if m:
            last_frame = max(last_frame, int(m.group(1)))

    matchframes = last_frame + 120
    out_header = rewrite_header(header, name, matchframes)

    out_lines: List[str] = []
    out_lines.extend(out_header)
    if out_lines and out_lines[-1].strip():
        out_lines.append("")
    for b in output_blocks:
        out_lines.extend(b)
        out_lines.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{name}.rpl"
    out_path.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")

    for dst_dir in [REPLAY_ROOT, REPLAY_PARKOUR]:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_path, dst_dir / out_path.name)

    summary = {
        "name": name,
        "source": str(source),
        "prelude": [0, PRELUDE_END],
        "loop": [LOOP_START, loop_end],
        "loop_len": loop_len,
        "loop_frames": len(loop_frames),
        "loop_repeats": LOOP_REPEATS,
        "last_frame": last_frame,
        "matchframes": matchframes,
        "output": str(out_path),
    }
    (OUT_DIR / f"{name}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    source = pick_source()
    print("Source:", source)
    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    header, blocks = split_blocks(lines)
    print("Frames source:", len(blocks), "min", min(blocks), "max", max(blocks))

    made = []
    for loop_end in LOOP_END_VARIANTS:
        p = build_variant(source, header, blocks, loop_end)
        made.append(p)
        print("Made:", p.name, f"loop={LOOP_START}->{loop_end}")

    print("\nÀ tester en priorité:")
    print("  xioi_loop_phase_v50_len255  (485->740)")
    print("  xioi_loop_phase_v50_len250  (485->735)")
    print("  xioi_loop_phase_v50_len245  (485->730)")
    print("\nCopies envoyées dans replay/ et replay/parkour/")


if __name__ == "__main__":
    main()
