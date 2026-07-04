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
HIDDEN = 128
EPOCHS = 250
BATCH_SIZE = 16
LR = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class SkillDataset(Dataset):
    def __init__(self, path):
        self.rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.rows.append(row)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]

        x = torch.tensor(row["state"], dtype=torch.long)
        y = torch.tensor(row["action"], dtype=torch.long)

        # valeurs Toribash 0..4
        return x, y


class SkillGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(NUM_CLASSES, 16)
        self.gru = nn.GRU(
            input_size=JOINTS * 16,
            hidden_size=HIDDEN,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(HIDDEN, JOINTS * NUM_CLASSES)

    def forward(self, x):
        # x: batch, seq, joints
        emb = self.embed(x)
        emb = emb.reshape(x.shape[0], x.shape[1], JOINTS * 16)

        out, _ = self.gru(emb)
        last = out[:, -1, :]

        logits = self.head(last)
        logits = logits.reshape(x.shape[0], JOINTS, NUM_CLASSES)
        return logits


def accuracy(logits, y):
    pred = logits.argmax(dim=-1)
    joint_acc = (pred == y).float().mean().item()
    exact_acc = (pred == y).all(dim=1).float().mean().item()
    return joint_acc, exact_acc


def train_one(skill_name):
    path = DATA_DIR / f"{skill_name}_skill_v1.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)

    dataset = SkillDataset(path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = SkillGRU().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    best_loss = 999999.0
    best_state = None

    print(f"\n=== Training {skill_name} ===")
    print(f"Dataset: {len(dataset)}")
    print(f"Device: {DEVICE}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        total_joint = 0.0
        total_exact = 0.0
        batches = 0

        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            logits = model(x)
            loss = loss_fn(logits.reshape(-1, NUM_CLASSES), y.reshape(-1))

            opt.zero_grad()
            loss.backward()
            opt.step()

            ja, ea = accuracy(logits.detach(), y)

            total_loss += loss.item()
            total_joint += ja
            total_exact += ea
            batches += 1

        avg_loss = total_loss / batches
        avg_joint = total_joint / batches
        avg_exact = total_exact / batches

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                "model": model.state_dict(),
                "skill": skill_name,
                "seq_len": SEQ_LEN,
                "joints": JOINTS,
                "num_classes": NUM_CLASSES,
                "hidden": HIDDEN,
                "rows": len(dataset),
                "best_loss": best_loss,
            }

        if epoch == 1 or epoch % 25 == 0 or epoch == EPOCHS:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={avg_loss:.4f} | "
                f"joint_acc={avg_joint:.4f} | "
                f"exact={avg_exact:.4f}"
            )

    out = MODEL_DIR / f"{skill_name}_gru_v1.pt"
    torch.save(best_state, out)
    print(f"[OK] Saved: {out}")
    print(f"Best loss: {best_loss:.4f}")


def main():
    print("=== ToribashAI Skill GRU Trainer V1 ===")
    for skill in ["launch", "walk"]:
        train_one(skill)


if __name__ == "__main__":
    main()
