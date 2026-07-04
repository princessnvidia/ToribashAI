#!/usr/bin/env python3
from __future__ import annotations
import json, random
from pathlib import Path
from collections import Counter
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path.home() / 'Documents' / 'ToribashAI'
DATASET = ROOT / 'datasets' / 'ml' / 'xioi_assassin_loop_v43_sequences.jsonl'
MODEL_OUT = ROOT / 'models' / 'xioi_assassin_loop_gru_v43.pt'
SUMMARY_OUT = ROOT / 'generated_replays' / 'xioi_assassin_loop_gru_v43_summary.json'

STATE_DIM = 100
ACTION_DIM = 20
SEQ_LEN = 8
HIDDEN = 192
LAYERS = 2
EPOCHS = 260
BATCH = 32
LR = 2e-3
SEED = 43

random.seed(SEED)
torch.manual_seed(SEED)

class WalkDataset(Dataset):
    def __init__(self, path: Path):
        self.rows = [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        return torch.tensor(r['seq'], dtype=torch.float32), torch.tensor(r['target'], dtype=torch.long)

class GRUAction(nn.Module):
    def __init__(self, state_dim=STATE_DIM, hidden=HIDDEN, layers=LAYERS):
        super().__init__()
        self.gru = nn.GRU(state_dim, hidden, num_layers=layers, batch_first=True, dropout=0.10 if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, ACTION_DIM * 5))
    def forward(self, x):
        y, _ = self.gru(x)
        z = y[:, -1]
        return self.head(z).view(-1, ACTION_DIM, 5)

@torch.no_grad()
def evaluate(model, ds, device):
    model.eval()
    xs = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    ys = torch.stack([ds[i][1] for i in range(len(ds))]).to(device)
    logits = model(xs)
    pred = logits.argmax(dim=-1)
    joint = (pred == ys).float().mean().item()
    exact = (pred == ys).all(dim=1).float().mean().item()
    nonzero_mask = ys != 0
    nonzero = (pred[nonzero_mask] == ys[nonzero_mask]).float().mean().item() if nonzero_mask.any() else 0.0
    return joint, exact, nonzero


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds = WalkDataset(DATASET)
    counts = Counter()
    for _, y in ds:
        counts.update(y.tolist())
    total = sum(counts.values())
    weights = []
    for c in range(5):
        freq = counts.get(c, 1) / max(1, total)
        weights.append((1.0 / (freq ** 0.35)))
    s = sum(weights) / len(weights)
    weights = torch.tensor([w / s for w in weights], dtype=torch.float32, device=device)

    loader = DataLoader(ds, batch_size=BATCH, shuffle=True)
    model = GRUAction().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    print('Device:', device)
    print('Dataset:', len(ds), 'state_dim', STATE_DIM, 'seq_len', SEQ_LEN)
    print('Class counts:', dict(counts))
    print('Class weights:', [round(float(x), 4) for x in weights])

    best_metric = -1.0
    best_epoch = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        losses = []
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits.reshape(-1, 5), y.reshape(-1))
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.item()))
        if epoch == 1 or epoch % 10 == 0 or epoch > EPOCHS - 5:
            joint, exact, nonzero = evaluate(model, ds, device)
            metric = joint + exact + nonzero
            print(f'Epoch {epoch:03d} | loss={sum(losses)/len(losses):.4f} joint={joint:.4f} exact={exact:.4f} nonzero={nonzero:.4f}')
            if metric > best_metric:
                best_metric = metric; best_epoch = epoch
                MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state': model.state_dict(),
                    'state_dim': STATE_DIM,
                    'seq_len': SEQ_LEN,
                    'hidden': HIDDEN,
                    'layers': LAYERS,
                    'dataset': str(DATASET),
                    'epoch': epoch,
                    'metric': metric,
                }, MODEL_OUT)
                print('  saved', MODEL_OUT)

    summary = {'version': 43, 'model': str(MODEL_OUT), 'best_epoch': best_epoch, 'best_metric': best_metric, 'rows': len(ds)}
    SUMMARY_OUT.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_OUT.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Done:', json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
