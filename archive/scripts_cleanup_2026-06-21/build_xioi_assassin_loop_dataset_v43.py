#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
from collections import Counter

ROOT = Path.home() / 'Documents' / 'ToribashAI'
REF = ROOT / 'generated_replays' / 'xioi_assassin_reference_v43_cut315.json'
OUT = ROOT / 'datasets' / 'ml' / 'xioi_assassin_loop_v43_sequences.jsonl'
SUMMARY = ROOT / 'generated_replays' / 'xioi_assassin_loop_v43_dataset_summary.json'

SEQ_LEN = 8
LOOP_START = 70
LOOP_END = 295   # deliberately before hip bug / no training after 315
CYCLES = 10
ACTION_DIM = 20
STATE_DIM = 100  # 20 joints * 5 one-hot states


def action_values(frame):
    vals = [0] * ACTION_DIM
    for j, v in frame.get('joints', []):
        if 0 <= int(j) < ACTION_DIM and 0 <= int(v) <= 4:
            vals[int(j)] = int(v)
    return vals


def onehot_action(vals):
    x = [0.0] * STATE_DIM
    for j, v in enumerate(vals):
        if 0 <= v <= 4:
            x[j * 5 + v] = 1.0
    return x


def main():
    ref = json.loads(REF.read_text(encoding='utf-8'))
    frames = [f for f in ref['frames'] if LOOP_START <= int(f['frame']) <= LOOP_END and f.get('joints')]
    frames.sort(key=lambda f: int(f['frame']))
    if len(frames) < SEQ_LEN + 2:
        raise RuntimeError('not enough loop frames')

    # make a clean cyclic action tape: no post-315 data, no hip-bug tail
    tape = []
    for c in range(CYCLES):
        for f in frames:
            vals = action_values(f)
            tape.append({
                'cycle': c,
                'source_frame': int(f['frame']),
                'state': onehot_action(vals),
                'values': vals,
            })

    rows = []
    counts = Counter()
    for i in range(0, len(tape) - SEQ_LEN - 1):
        seq = [tape[i+k]['state'] for k in range(SEQ_LEN)]
        target = tape[i+SEQ_LEN]['values']
        for v in target:
            counts[v] += 1
        rows.append({
            'seq': seq,
            'target': target,
            'source_frames': [tape[i+k]['source_frame'] for k in range(SEQ_LEN)],
            'target_source_frame': tape[i+SEQ_LEN]['source_frame'],
            'cycle': tape[i+SEQ_LEN]['cycle'],
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')

    summary = {
        'version': 43,
        'reference': str(REF),
        'dataset': str(OUT),
        'rows': len(rows),
        'seq_len': SEQ_LEN,
        'state_dim': STATE_DIM,
        'action_dim': ACTION_DIM,
        'loop_start': LOOP_START,
        'loop_end': LOOP_END,
        'cycles': CYCLES,
        'unique_loop_action_frames': len(frames),
        'value_counts': counts.most_common(),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
