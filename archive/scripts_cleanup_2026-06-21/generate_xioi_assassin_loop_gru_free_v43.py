#!/usr/bin/env python3
from __future__ import annotations
import json, re, shutil
from pathlib import Path
import torch
from torch import nn

ROOT = Path.home() / 'Documents' / 'ToribashAI'
DATASET = ROOT / 'datasets' / 'ml' / 'xioi_assassin_loop_v43_sequences.jsonl'
MODEL_PATH = ROOT / 'models' / 'xioi_assassin_loop_gru_v43.pt'
TEMPLATE = ROOT / 'generated_replays' / 'xioi_427_assassincreedhunter_v37.rpl'
OUT = ROOT / 'generated_replays' / 'xioi_assassin_loop_gru_free_v43.rpl'
ACTIONS_OUT = ROOT / 'generated_replays' / 'xioi_assassin_loop_gru_free_v43_actions.json'
SUMMARY_OUT = ROOT / 'generated_replays' / 'xioi_assassin_loop_gru_free_v43_summary.json'
STEAM_REPLAY = Path.home() / '.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay'
STEAM_PARKOUR = STEAM_REPLAY / 'parkour'

STATE_DIM = 100
ACTION_DIM = 20
SEQ_LEN = 8
CUT_AFTER_FRAME = 315
GENERATE_TO_FRAME = 1000
TURNFRAMES = 5
FIGHTNAME = 'xioi_assassin_loop_gru_free_v43'

FRAME_RE = re.compile(r'^FRAME\s+(\d+);')
JOINT0_RE = re.compile(r'^JOINT\s+0;')

class GRUAction(nn.Module):
    def __init__(self, state_dim=STATE_DIM, hidden=192, layers=2):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden, num_layers=layers, batch_first=True, dropout=0.10 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, ACTION_DIM * 5))
    def forward(self, x):
        y, _ = self.gru(x)
        z = y[:, -1]
        return self.head(z).view(-1, ACTION_DIM, 5)

def onehot(vals):
    x = [0.0] * (ACTION_DIM * 5)
    for j, v in enumerate(vals):
        if 0 <= int(v) <= 4:
            x[j*5 + int(v)] = 1.0
    return x

def pairs_from_vals(vals):
    return [[j, int(v)] for j, v in enumerate(vals) if int(v) != 0]

def load_model(device):
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = GRUAction(state_dim=ckpt.get('state_dim', STATE_DIM), hidden=ckpt.get('hidden', 192), layers=ckpt.get('layers', 2)).to(device)
    model.load_state_dict(ckpt['model_state'], strict=True)
    model.eval()
    print('Loaded:', MODEL_PATH, 'epoch=', ckpt.get('epoch'))
    return model, ckpt

@torch.no_grad()
def generate_actions():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rows = [json.loads(l) for l in DATASET.read_text(encoding='utf-8').splitlines() if l.strip()]
    model, ckpt = load_model(device)
    # seed with first sequence from clean loop dataset, not launch frames and not hip-bug tail
    seq_states = rows[0]['seq'][:]
    actions = []
    frame = CUT_AFTER_FRAME + TURNFRAMES
    while frame <= GENERATE_TO_FRAME:
        x = torch.tensor([seq_states[-SEQ_LEN:]], dtype=torch.float32, device=device)
        pred = model(x).argmax(dim=-1)[0].cpu().tolist()
        actions.append({'frame': frame, 'values': pred, 'pairs': pairs_from_vals(pred)})
        seq_states.append(onehot(pred))
        frame += TURNFRAMES
    return actions, ckpt

def make_replay(generated_actions):
    lines = TEMPLATE.read_text(encoding='utf-8', errors='ignore').splitlines()
    out = []
    cur_frame = None
    skip_joint0 = False
    for line in lines:
        m = FRAME_RE.match(line.strip())
        if m:
            cur_frame = int(m.group(1))
            if cur_frame > CUT_AFTER_FRAME:
                break
            out.append(line)
            skip_joint0 = False
            continue
        if line.startswith('FIGHTNAME 0;'):
            out.append(f'FIGHTNAME 0; {FIGHTNAME}')
            continue
        if cur_frame is not None and cur_frame <= CUT_AFTER_FRAME and JOINT0_RE.match(line.strip()):
            # keep original JOINTs until 315 exactly; no rewrite in the safe copied part
            out.append(line)
            continue
        out.append(line)

    if out and out[-1].strip() != '':
        out.append('')
    for a in generated_actions:
        out.append(f'FRAME {a["frame"]}; 0 0 0 0')
        if a['pairs']:
            flat = ' '.join(f'{j} {v}' for j, v in a['pairs'])
            out.append(f'JOINT 0; {flat}')
        out.append('')
    return '\n'.join(out) + '\n'

def main():
    actions, ckpt = generate_actions()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(make_replay(actions), encoding='utf-8')
    ACTIONS_OUT.write_text(json.dumps({'version':43, 'actions':actions}, indent=2), encoding='utf-8')
    for dst_dir in [STEAM_REPLAY, STEAM_PARKOUR]:
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUT, dst_dir / OUT.name)
        print('Copied to:', dst_dir / OUT.name)
    summary = {
        'version': 43,
        'mode': 'strict_loop_only_free_run',
        'cut_after_frame': CUT_AFTER_FRAME,
        'generated_to_frame': GENERATE_TO_FRAME,
        'generated_actions': len(actions),
        'model': str(MODEL_PATH),
        'dataset': str(DATASET),
        'rpl': str(OUT),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
