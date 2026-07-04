#!/usr/bin/env python3
from __future__ import annotations
import json, re
from pathlib import Path
from collections import Counter

ROOT = Path.home() / 'Documents' / 'ToribashAI'
SRC_RPL = ROOT / 'generated_replays' / 'xioi_427_assassincreedhunter_v37.rpl'
OUT = ROOT / 'generated_replays' / 'xioi_assassin_reference_v43_cut315.json'
SUMMARY = ROOT / 'generated_replays' / 'xioi_assassin_reference_v43_summary.json'

MAX_KEEP_FRAME = 315
TRAIN_SAFE_END = 295  # avoid the small hip bug area around 300 during loop learning

FRAME_RE = re.compile(r'^FRAME\s+(\d+);')
JOINT_RE = re.compile(r'^JOINT\s+(\d+);\s*(.*)$')
NUM_RE = re.compile(r'-?\d+(?:\.\d+)?')

POINTS = {
    'head': 0,
    'chest': 1,
    'lumbar': 2,
    'abs': 3,
    'left_shoulder': 5,
    'right_shoulder': 8,
    'left_hip': 13,
    'right_hip': 14,
    'left_foot': 19,
    'right_foot': 20,
}


def parse_joint_pairs(rest: str):
    vals = [int(x) for x in re.findall(r'-?\d+', rest)]
    if len(vals) < 2:
        return []
    out = []
    for i in range(0, len(vals) - 1, 2):
        j, v = vals[i], vals[i + 1]
        if 0 <= j <= 19 and 0 <= v <= 4:
            out.append([j, v])
    return out


def parse_vecs(rest: str):
    nums = [float(x) for x in NUM_RE.findall(rest)]
    return [nums[i:i+3] for i in range(0, len(nums) - 2, 3)]


def main():
    if not SRC_RPL.exists():
        raise FileNotFoundError(SRC_RPL)
    frames = {}
    meta_lines = []
    cur = None
    for line in SRC_RPL.read_text(encoding='utf-8', errors='ignore').splitlines():
        m = FRAME_RE.match(line.strip())
        if m:
            cur = int(m.group(1))
            if cur <= MAX_KEEP_FRAME:
                frames.setdefault(str(cur), {'frame': cur, 'joints': [], 'points': {}})
            continue
        if cur is None:
            meta_lines.append(line)
            continue
        if cur > MAX_KEEP_FRAME:
            continue
        s = line.strip()
        jm = JOINT_RE.match(s)
        if jm and int(jm.group(1)) == 0:
            pairs = parse_joint_pairs(jm.group(2))
            if pairs:
                frames[str(cur)]['joints'].extend(pairs)
            continue
        if s.startswith('POS 0;'):
            vecs = parse_vecs(s.split(';',1)[1])
            for name, idx in POINTS.items():
                if idx < len(vecs):
                    frames[str(cur)]['points'][name] = vecs[idx]

    ordered = [frames[k] for k in sorted(frames, key=lambda x: int(x))]
    counts = Counter()
    for f in ordered:
        for _, v in f['joints']:
            counts[v] += 1

    data = {
        'version': 43,
        'description': 'Strict Xioi assassin reference cut to frame 315; loop training should ignore >=295 to avoid hip bug.',
        'source_rpl': str(SRC_RPL),
        'max_keep_frame': MAX_KEEP_FRAME,
        'train_safe_end': TRAIN_SAFE_END,
        'frames': ordered,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
    summary = {
        'version': 43,
        'source': str(SRC_RPL),
        'reference': str(OUT),
        'frame_count': len(ordered),
        'frame_min': ordered[0]['frame'] if ordered else None,
        'frame_max': ordered[-1]['frame'] if ordered else None,
        'train_safe_end': TRAIN_SAFE_END,
        'joint_value_counts': counts.most_common(),
        'joint_frames': sum(1 for f in ordered if f['joints']),
        'pos_frames': sum(1 for f in ordered if f['points']),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
