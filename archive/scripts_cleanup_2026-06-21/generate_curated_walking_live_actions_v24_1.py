#!/usr/bin/env python3
"""
generate_curated_walking_live_actions_v24_1.py

Fix V24.1:
- reprend exactement l'architecture de train_curated_walking_gru_v23_1.py
- checkpoint: GRUAction -> LayerNorm(HIDDEN) -> Linear(HIDDEN, 20*5)
- génère un JSON d'actions live lisible par toribash_curated_walking_gru_live_runner_v24.lua
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch import nn


ROOT = Path.home() / "Documents" / "ToribashAI"
DATASET_PATH = ROOT / "datasets" / "ml" / "curated_walking_v23_1_sequences.jsonl"
MODEL_PATH = ROOT / "models" / "curated_walking_gru_v23_1.pt"

OUT_DIR = ROOT / "generated_replays"
OUT_ACTIONS = OUT_DIR / "curated_walking_gru_v24_1_live_actions.json"

TORIBASH_SCRIPT_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"
)
TORIBASH_ACTIONS = TORIBASH_SCRIPT_DIR / "curated_walking_gru_v24_live_actions_current.json"

GENERATED_STEPS = 180
TURNFRAMES = 5
MAX_ACTIVE_JOINTS = 7
MIN_ACTIVE_JOINTS = 3

# Dataset V23.1: seq_len=8, state_dim=42, action_dim=20.
# On garde une température douce pour éviter que tout devienne 0.
VALUE_TEMPERATURE = 0.95


class GRUAction(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 128, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 20 * 5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        z = y[:, -1, :]
        return self.head(z).view(-1, 20, 5)


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"Dataset vide: {path}")
    return rows


def get_state_seq(row: dict[str, Any]) -> list[list[float]]:
    for key in ("state_seq", "states", "x"):
        if key in row:
            return row[key]
    raise KeyError(f"Impossible de trouver state_seq dans row keys={list(row.keys())}")


def get_action(row: dict[str, Any]) -> list[int] | None:
    for key in ("action", "target", "y", "actions"):
        if key in row and isinstance(row[key], list):
            vals = row[key]
            if len(vals) == 20:
                return [int(v) for v in vals]
    return None


def choose_seed(rows: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    # Choisit une séquence avec un peu d'activité, plutôt au milieu du début de marche.
    candidates: list[tuple[int, dict[str, Any], int]] = []
    for i, row in enumerate(rows):
        action = get_action(row)
        active = sum(1 for v in action or [] if int(v) != 0)
        frame = int(row.get("target_frame", row.get("frame", 0)) or 0)
        if 2 <= active <= 9 and 35 <= frame <= 220:
            candidates.append((i, row, active))

    if candidates:
        # Milieu de liste = évite le tout début figé et les frames trop tardives.
        i, row, _ = candidates[len(candidates) // 2]
        return i, row

    return len(rows) // 2, rows[len(rows) // 2]


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict):
        return ckpt
    raise RuntimeError("Checkpoint inattendu: pas un dict")


def get_model_state(ckpt: dict[str, Any]) -> dict[str, torch.Tensor]:
    for key in ("model_state", "model_state_dict", "state_dict"):
        if key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    # parfois torch.save(model.state_dict()) directement
    if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt  # type: ignore[return-value]
    raise KeyError(f"Impossible de trouver le state_dict dans checkpoint keys={list(ckpt.keys())}")


def infer_hidden(model_state: dict[str, torch.Tensor], ckpt: dict[str, Any]) -> int:
    if "hidden" in ckpt:
        return int(ckpt["hidden"])
    if "gru.weight_hh_l0" in model_state:
        # GRU weight_hh_l0 shape = [3*hidden, hidden]
        return int(model_state["gru.weight_hh_l0"].shape[1])
    if "head.1.weight" in model_state:
        return int(model_state["head.1.weight"].shape[1])
    return 128


def load_model(device: torch.device, state_dim: int) -> GRUAction:
    ckpt = load_checkpoint(MODEL_PATH, device)
    model_state = get_model_state(ckpt)
    hidden = infer_hidden(model_state, ckpt)
    num_layers = int(ckpt.get("num_layers", 1)) if isinstance(ckpt, dict) else 1

    model = GRUAction(state_dim=state_dim, hidden=hidden, num_layers=num_layers).to(device)
    model.load_state_dict(model_state, strict=True)
    model.eval()

    print("Loaded model:", MODEL_PATH)
    print("Hidden:", hidden, "layers:", num_layers)
    return model


def predict_action(model: GRUAction, state_seq: list[list[float]], device: torch.device) -> tuple[list[list[int]], list[int]]:
    x = torch.tensor([state_seq], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x)[0] / VALUE_TEMPERATURE  # [20, 5]
        probs = torch.softmax(logits, dim=-1)
        values = torch.argmax(probs, dim=-1).detach().cpu().tolist()  # 0..4
        nonzero_score = (1.0 - probs[:, 0]).detach().cpu().tolist()

    ranked = sorted(range(20), key=lambda j: float(nonzero_score[j]), reverse=True)

    pairs: list[list[int]] = []
    for j in ranked:
        v = int(values[j])
        if v != 0:
            pairs.append([int(j), v])
        if len(pairs) >= MAX_ACTIVE_JOINTS:
            break

    # Si le modèle prédit trop de 0, on force les meilleurs nonzero logits.
    if len(pairs) < MIN_ACTIVE_JOINTS:
        pairs = []
        for j in ranked[:MIN_ACTIVE_JOINTS]:
            # meilleur état non-zéro entre 1..4
            nonzero_logits = logits[j, 1:]
            v = int(torch.argmax(nonzero_logits).item()) + 1
            pairs.append([int(j), v])

    return pairs, [int(v) for v in values]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Dataset:", DATASET_PATH)

    rows = load_rows(DATASET_PATH)
    seed_idx, seed = choose_seed(rows)
    state_seq = get_state_seq(seed)
    state_dim = len(state_seq[0])

    print("Rows:", len(rows))
    print("Seed index:", seed_idx)
    print("Seed replay:", seed.get("replay") or seed.get("source") or seed.get("file"))
    print("Seed frame:", seed.get("target_frame") or seed.get("frame"))
    print("State dim:", state_dim)

    model = load_model(device, state_dim)

    actions: list[dict[str, Any]] = []
    for step in range(GENERATED_STEPS):
        pairs, values = predict_action(model, state_seq, device)
        actions.append({
            "frame": step * TURNFRAMES,
            "pairs": pairs,
            "values": values,
        })
        # Génération ouverte: sans simulateur Python on garde le même contexte.
        # La boucle live Toribash teste surtout si le modèle produit un vocabulaire walking.

    data = {
        "name": "curated_walking_gru_v24_1_live_actions",
        "version": "24.1",
        "model": str(MODEL_PATH),
        "dataset": str(DATASET_PATH),
        "turnframes": TURNFRAMES,
        "generated_steps": GENERATED_STEPS,
        "seed_index": seed_idx,
        "seed_frame": seed.get("target_frame") or seed.get("frame"),
        "actions": actions,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ACTIONS.write_text(json.dumps(data, indent=2), encoding="utf-8")

    TORIBASH_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_ACTIONS, TORIBASH_ACTIONS)

    print("Actions projet:", OUT_ACTIONS)
    print("Actions Steam:", TORIBASH_ACTIONS)
    print("First action:", actions[0])


if __name__ == "__main__":
    main()
