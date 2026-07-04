#!/usr/bin/env python3
"""
generate_curated_walking_live_actions_v24.py

Génère une séquence d'actions depuis le GRU walking spécialisé V23.1.
Entrée modèle:
  models/curated_walking_gru_v23_1.pt
Entrée dataset seed:
  datasets/ml/curated_walking_v23_sequences.jsonl
  ou datasets/ml/curated_walking_v23_1_sequences.jsonl
Sortie:
  generated_replays/curated_walking_gru_v24_live_actions.json
  data/script/curated_walking_gru_v24_live_actions_current.json
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ROOT = Path.home() / "Documents" / "ToribashAI"
MODEL_PATH = ROOT / "models" / "curated_walking_gru_v23_1.pt"
DATASET_CANDIDATES = [
    ROOT / "datasets" / "ml" / "curated_walking_v23_1_sequences.jsonl",
    ROOT / "datasets" / "ml" / "curated_walking_v23_sequences.jsonl",
]
OUT_DIR = ROOT / "generated_replays"
OUT_JSON = OUT_DIR / "curated_walking_gru_v24_live_actions.json"

TORIBASH_SCRIPT_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"
)
TORIBASH_JSON = TORIBASH_SCRIPT_DIR / "curated_walking_gru_v24_live_actions_current.json"

GENERATED_STEPS = 180
TURNFRAMES = 5
MAX_ACTIVE_JOINTS = 8
MIN_ACTIVE_JOINTS = 2
NONZERO_THRESHOLD = 0.36
TEMPERATURE = 0.85


class WalkingGRU(nn.Module):
    def __init__(self, state_dim: int, hidden_size: int = 128, num_layers: int = 1, action_dim: int = 20, classes: int = 5):
        super().__init__()
        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim * classes),
        )
        self.action_dim = action_dim
        self.classes = classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        last = y[:, -1, :]
        logits = self.head(last)
        return logits.view(-1, self.action_dim, self.classes)


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


def find_dataset() -> Path:
    for p in DATASET_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError("Aucun dataset curated walking trouvé: " + ", ".join(str(p) for p in DATASET_CANDIDATES))


def get_state_seq(row: dict[str, Any]) -> list[list[float]]:
    for key in ("state_seq", "states", "x"):
        if key in row:
            return row[key]
    raise KeyError("Impossible de trouver state_seq/states/x dans une ligne dataset")


def get_action(row: dict[str, Any]) -> list[int]:
    for key in ("action", "actions", "target", "y"):
        if key in row:
            return [int(v) for v in row[key]]
    return [0] * 20


def choose_seed(rows: list[dict[str, Any]]) -> tuple[int, dict[str, Any]]:
    scored = []
    for i, row in enumerate(rows):
        action = get_action(row)
        nonzero = sum(1 for v in action if int(v) != 0)
        frame = int(row.get("target_frame", row.get("frame", 0)) or 0)
        # On préfère un début actif mais pas explosif.
        score = nonzero * 10 - abs(frame - 80) * 0.03
        if 2 <= nonzero <= 9:
            scored.append((score, i, row))
    if not scored:
        return 0, rows[0]
    scored.sort(reverse=True)
    _, i, row = scored[min(3, len(scored) - 1)]
    return i, row


def load_model(device: torch.device, state_dim: int) -> WalkingGRU:
    ckpt = torch.load(MODEL_PATH, map_location=device)
    ckpt_state_dim = int(ckpt.get("state_dim", state_dim)) if isinstance(ckpt, dict) else state_dim
    hidden = int(ckpt.get("hidden_size", 128)) if isinstance(ckpt, dict) else 128
    layers = int(ckpt.get("num_layers", 1)) if isinstance(ckpt, dict) else 1
    model = WalkingGRU(ckpt_state_dim, hidden_size=hidden, num_layers=layers)
    state = ckpt.get("model_state", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def predict(model: WalkingGRU, state_seq: list[list[float]], device: torch.device) -> tuple[list[list[int]], list[list[float]]]:
    x = torch.tensor([state_seq], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = model(x)[0] / TEMPERATURE
        probs = torch.softmax(logits, dim=-1)
    probs_l = probs.detach().cpu().tolist()

    nonzero_scores = []
    values = []
    for j in range(20):
        p = probs_l[j]
        best_v = max(range(1, 5), key=lambda v: p[v])
        values.append(best_v)
        nonzero_scores.append(1.0 - p[0])

    ranked = sorted(range(20), key=lambda j: nonzero_scores[j], reverse=True)
    active = [j for j in ranked if nonzero_scores[j] >= NONZERO_THRESHOLD][:MAX_ACTIVE_JOINTS]
    if len(active) < MIN_ACTIVE_JOINTS:
        active = ranked[:MIN_ACTIVE_JOINTS]

    pairs = [[int(j), int(values[j])] for j in active]
    top = [[int(j), float(nonzero_scores[j])] for j in ranked[:8]]
    return pairs, top


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = find_dataset()
    rows = load_rows(dataset_path)
    seed_idx, seed = choose_seed(rows)
    state_seq = get_state_seq(seed)
    state_dim = len(state_seq[0])

    print("Device:", device)
    print("Dataset:", dataset_path)
    print("Rows:", len(rows))
    print("Seed index:", seed_idx)
    print("Seed replay:", seed.get("replay") or seed.get("source") or seed.get("file"))
    print("Seed frame:", seed.get("target_frame", seed.get("frame")))
    print("State dim:", state_dim)

    model = load_model(device, state_dim)

    actions = []
    for step in range(GENERATED_STEPS):
        pairs, top = predict(model, state_seq, device)
        actions.append({
            "frame": step * TURNFRAMES,
            "pairs": pairs,
            "top_nonzero": top,
        })
        # V24 open-loop: on boucle le même contexte latent.
        # Le live runner prouvera déjà si le modèle spécialisé marche est plus doux.

    data = {
        "name": "curated_walking_gru_v24_live_actions",
        "version": "24",
        "model": str(MODEL_PATH),
        "dataset": str(dataset_path),
        "turnframes": TURNFRAMES,
        "generated_steps": GENERATED_STEPS,
        "seed_index": seed_idx,
        "seed_replay": seed.get("replay") or seed.get("source") or seed.get("file"),
        "seed_frame": seed.get("target_frame", seed.get("frame")),
        "actions": actions,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    TORIBASH_SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(OUT_JSON, TORIBASH_JSON)

    print("Actions projet:", OUT_JSON)
    print("Actions Steam:", TORIBASH_JSON)


if __name__ == "__main__":
    main()
