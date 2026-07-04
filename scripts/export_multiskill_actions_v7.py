#!/usr/bin/env python3
import json
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path.home() / "Documents" / "ToribashAI"
DATA_DIR = ROOT / "datasets" / "skills"
MODEL_DIR = ROOT / "models" / "skills"
OUT = ROOT / "evolution" / "multiskill_actions_v7.json"

SEQ_LEN = 8
JOINTS = 20
NUM_CLASSES = 5
HIDDEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        emb = self.embed(x)
        emb = emb.reshape(x.shape[0], x.shape[1], JOINTS * 16)
        out, _ = self.gru(emb)
        last = out[:, -1, :]
        logits = self.head(last)
        return logits.reshape(x.shape[0], JOINTS, NUM_CLASSES)


def load_rows(skill):
    path = DATA_DIR / f"{skill}_skill_v1.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def load_model(skill):
    path = MODEL_DIR / f"{skill}_gru_v1.pt"
    ckpt = torch.load(path, map_location=DEVICE)

    model = SkillGRU().to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, ckpt


def export_skill(skill):
    rows = load_rows(skill)
    model, ckpt = load_model(skill)

    actions = []
    true_actions = []
    exact = 0
    total = 0
    joint_ok = 0
    joint_total = 0

    with torch.no_grad():
        for idx, row in enumerate(rows):
            x = torch.tensor([row["state"]], dtype=torch.long, device=DEVICE)
            y = row["action"]

            logits = model(x)
            pred = logits.argmax(dim=-1)[0].cpu().tolist()

            actions.append({
                "step": idx,
                "target_frame": row["target_frame"],
                "joints": pred,
            })

            true_actions.append({
                "step": idx,
                "target_frame": row["target_frame"],
                "joints": y,
            })

            if pred == y:
                exact += 1

            for a, b in zip(pred, y):
                if a == b:
                    joint_ok += 1
                joint_total += 1

            total += 1

    return {
        "skill": skill,
        "rows": len(rows),
        "model_best_loss": ckpt.get("best_loss"),
        "joint_accuracy_on_skill_dataset": joint_ok / max(1, joint_total),
        "exact_accuracy_on_skill_dataset": exact / max(1, total),
        "actions": actions,
        "true_actions": true_actions,
    }


def main():
    print("=== Export multiskill actions V7 ===")
    print(f"Device: {DEVICE}")

    out = {
        "name": "toribashai_multiskill_actions_v7",
        "mod": "ToribashAI/toribashai_xioi_city_v1.tbm",
        "seq_len": SEQ_LEN,
        "joints": list(range(JOINTS)),
        "controller": [
            {"skill": "launch", "repeat": 1},
            {"skill": "walk", "repeat": "loop"},
        ],
        "skills": {
            "launch": export_skill("launch"),
            "walk": export_skill("walk"),
        },
    }

    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[OK] écrit: {OUT}")

    print("launch acc:", out["skills"]["launch"]["joint_accuracy_on_skill_dataset"], out["skills"]["launch"]["exact_accuracy_on_skill_dataset"])
    print("walk acc:", out["skills"]["walk"]["joint_accuracy_on_skill_dataset"], out["skills"]["walk"]["exact_accuracy_on_skill_dataset"])


if __name__ == "__main__":
    main()
