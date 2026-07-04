#!/usr/bin/env python3
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path.home() / "Documents" / "ToribashAI"
DATA_DIR = ROOT / "datasets" / "skills"
MODEL_DIR = ROOT / "models" / "skills"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 8
JOINTS = 20
NUM_CLASSES = 5

# joints 20 + pos 63 + qat 84 + linvel 63 + angvel 63
STATE_DIM = 293

HIDDEN = 192
EPOCHS = 200
BATCH_SIZE = 32
LR = 8e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SkillPhysicsDataset(Dataset):
    def __init__(self, path):
        self.rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                self.rows.append(json.loads(line))

    def __len__(self):
        return len(self.rows)

    def vectorize_frame(self, frame):
        joints = [v / 4.0 for v in frame["joints"]]
        pos = [v / 50.0 for v in frame["pos"]]
        qat = frame["qat"]
        linvel = [v / 50.0 for v in frame["linvel"]]
        angvel = [v / 50.0 for v in frame["angvel"]]
        return joints + pos + qat + linvel + angvel

    def __getitem__(self, idx):
        row = self.rows[idx]

        x = torch.tensor(
            [self.vectorize_frame(f) for f in row["state"]],
            dtype=torch.float32,
        )

        y = torch.tensor(row["action"], dtype=torch.long)
        return x, y


class SkillPhysicsGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(
            input_size=STATE_DIM,
            hidden_size=HIDDEN,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN),
            nn.Linear(HIDDEN, 256),
            nn.GELU(),
            nn.Linear(256, JOINTS * NUM_CLASSES),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits.reshape(x.shape[0], JOINTS, NUM_CLASSES)


def acc(logits, y):
    pred = logits.argmax(dim=-1)
    joint = (pred == y).float().mean().item()
    exact = (pred == y).all(dim=1).float().mean().item()
    return joint, exact


def train(skill):
    path = DATA_DIR / f"{skill}_skill_v2.jsonl"
    dataset = SkillPhysicsDataset(path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SkillPhysicsGRU().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best = None
    best_loss = 999999.0

    print(f"\n=== Train {skill} physics GRU V2 ===")
    print(f"Rows: {len(dataset)}")
    print(f"Device: {DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0
        total_joint = 0
        total_exact = 0
        batches = 0

        model.train()
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = loss_fn(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            j, e = acc(logits.detach(), y)

            total_loss += loss.item()
            total_joint += j
            total_exact += e
            batches += 1

        avg_loss = total_loss / batches
        avg_joint = total_joint / batches
        avg_exact = total_exact / batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best = {
                "model": model.state_dict(),
                "skill": skill,
                "seq_len": SEQ_LEN,
                "state_dim": STATE_DIM,
                "hidden": HIDDEN,
                "joints": JOINTS,
                "num_classes": NUM_CLASSES,
                "rows": len(dataset),
                "best_loss": best_loss,
            }

        if epoch == 1 or epoch % 20 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={avg_loss:.5f} | "
                f"joint={avg_joint:.4f} | "
                f"exact={avg_exact:.4f}"
            )

    out = MODEL_DIR / f"{skill}_physics_gru_v2.pt"
    torch.save(best, out)

    print(f"[OK] saved {out}")
    print(f"Best loss: {best_loss:.5f}")


def main():
    print("=== ToribashAI Skill Physics GRU V2 ===")
    train("walk")
    train("launch")


if __name__ == "__main__":
    main()
