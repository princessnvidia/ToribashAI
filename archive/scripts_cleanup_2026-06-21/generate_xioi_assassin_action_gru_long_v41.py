#!/usr/bin/env python3
"""
generate_xioi_assassin_action_gru_long_v41.py

V41 = long free-running test for the Xioi assassin walking action-GRU.

It uses the action-only GRU trained by V40 if present, or trains it again from
xioi_assassin_walk_v38_sequences.jsonl.

Outputs long RPLs:
  - seed35  -> 1000 frames
  - seed140 -> 1000 frames
  - seed200 -> 1200 frames

Important:
  Up to the end of the original template, the script preserves the full RPL
  structure (POS/QAT/LINVEL/ANGVEL) and only replaces JOINT 0 lines.
  After the original template ends, it appends FRAME + JOINT commands so
  Toribash continues the simulation under physics.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path.home() / "Documents" / "ToribashAI"
OUT_DIR = ROOT / "generated_replays"
DATASET_JSONL = ROOT / "datasets/ml/xioi_assassin_walk_v38_sequences.jsonl"
TEMPLATE_RPL = OUT_DIR / "xioi_427_assassincreedhunter_v37.rpl"
MODEL_PATH = ROOT / "models/xioi_assassin_action_gru_free_v40.pt"
SUMMARY_PATH = OUT_DIR / "xioi_assassin_action_gru_long_v41_summary.json"

STEAM_REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay"
)
STEAM_PARKOUR_DIR = STEAM_REPLAY_DIR / "parkour"

SEQ_LEN = 8
HIDDEN = 192
LAYERS = 2
EPOCHS = 550
LR = 2e-3
STEP_FRAMES = 5
RUNS = [
    {"seed_frame": 35, "target_frame": 1000},
    {"seed_frame": 140, "target_frame": 1000},
    {"seed_frame": 200, "target_frame": 1200},
]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActionGRU(nn.Module):
    def __init__(self, input_dim: int = 100, hidden: int = HIDDEN, layers: int = LAYERS):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 20 * 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        z = y[:, -1, :]
        return self.head(z).view(-1, 20, 5)


def one_hot_action(action: list[int]) -> list[float]:
    vals: list[float] = []
    for v in action:
        vv = int(v)
        if vv < 0 or vv > 4:
            vv = 0
        vals.extend(1.0 if c == vv else 0.0 for c in range(5))
    return vals


def row_action(row: dict[str, Any]) -> list[int]:
    for key in ("action", "target", "y", "values"):
        if key in row:
            vals = [int(v) for v in row[key]]
            if len(vals) == 20:
                return vals
    raise KeyError(f"No action key in row keys={list(row.keys())}")


def row_frame(row: dict[str, Any], fallback: int) -> int:
    for key in ("target_frame", "frame", "next_frame", "action_frame"):
        if key in row:
            return int(row[key])
    return fallback


def load_actions_from_dataset() -> list[tuple[int, list[int]]]:
    if not DATASET_JSONL.exists():
        raise FileNotFoundError(DATASET_JSONL)
    items: dict[int, list[int]] = {}
    with DATASET_JSONL.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            fr = row_frame(row, fallback=idx * STEP_FRAMES)
            items[fr] = row_action(row)
    if not items:
        raise RuntimeError("No actions loaded from dataset")
    return sorted(items.items(), key=lambda x: x[0])


def build_training_pairs(actions: list[tuple[int, list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[list[list[float]]] = []
    ys: list[list[int]] = []
    only_actions = [a for _, a in actions]
    for i in range(SEQ_LEN, len(only_actions)):
        xs.append([one_hot_action(a) for a in only_actions[i - SEQ_LEN : i]])
        ys.append(only_actions[i])
    if not xs:
        raise RuntimeError("Not enough actions for sequence training")
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def class_weights(y: torch.Tensor) -> torch.Tensor:
    counts = Counter(int(v) for v in y.flatten().tolist())
    total = sum(counts.values())
    weights = []
    for c in range(5):
        freq = counts.get(c, 1) / max(1, total)
        weights.append((1.0 / max(freq, 1e-6)) ** 0.35)
    mean_w = sum(weights) / len(weights)
    weights = [w / mean_w for w in weights]
    print("Class counts:", dict(counts))
    print("Class weights:", [round(w, 4) for w in weights])
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def evaluate(model: ActionGRU, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x.to(DEVICE))
        pred = torch.argmax(logits, dim=-1).cpu()
    correct = (pred == y).float()
    exact_rows = (correct.mean(dim=1) == 1.0).float()
    nz_mask = y != 0
    return {
        "joint": float(correct.mean().item()),
        "exact": float(exact_rows.mean().item()),
        "nonzero": float((pred[nz_mask] == y[nz_mask]).float().mean().item()) if nz_mask.any() else 0.0,
    }


def train_action_gru(actions: list[tuple[int, list[int]]]) -> ActionGRU:
    x, y = build_training_pairs(actions)
    model = ActionGRU().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    weights = class_weights(y)
    x_dev = x.to(DEVICE)
    y_dev = y.to(DEVICE)

    best_metric = -1.0
    best_state: dict[str, torch.Tensor] | None = None

    print("Device:", DEVICE)
    print("Action rows:", len(actions), "train pairs:", len(x), "seq_len:", SEQ_LEN)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(x_dev)
        loss = F.cross_entropy(logits.reshape(-1, 5), y_dev.reshape(-1), weight=weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if epoch == 1 or epoch % 10 == 0 or epoch > EPOCHS - 5:
            m = evaluate(model, x, y)
            metric = m["joint"] + m["exact"] + m["nonzero"]
            print(
                f"Epoch {epoch:03d} | loss={loss.item():.4f} "
                f"joint={m['joint']:.4f} exact={m['exact']:.4f} nonzero={m['nonzero']:.4f}"
            )
            if metric > best_metric:
                best_metric = metric
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": best_state,
                        "seq_len": SEQ_LEN,
                        "hidden": HIDDEN,
                        "layers": LAYERS,
                        "input_dim": 100,
                        "epoch": epoch,
                        "metric": metric,
                        "source": str(DATASET_JSONL),
                    },
                    MODEL_PATH,
                )
                print("  saved", MODEL_PATH)

    if best_state:
        model.load_state_dict(best_state)
    return model


def load_or_train_model(actions: list[tuple[int, list[int]]]) -> ActionGRU:
    if MODEL_PATH.exists():
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
        hidden = int(ckpt.get("hidden", HIDDEN))
        layers = int(ckpt.get("layers", LAYERS))
        model = ActionGRU(hidden=hidden, layers=layers).to(DEVICE)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state, strict=True)
        model.eval()
        print("Loaded action-GRU:", MODEL_PATH, "hidden=", hidden, "layers=", layers, "epoch=", ckpt.get("epoch"))
        return model
    print("No V40 action-GRU model found, training one now...")
    return train_action_gru(actions)


def parse_frame_number(line: str) -> int | None:
    m = re.match(r"^FRAME\s+(-?\d+)\s*;", line.strip())
    return int(m.group(1)) if m else None


def is_joint0_line(line: str) -> bool:
    return line.lstrip().startswith("JOINT 0;")


def compact_joint_line(player: int, action: list[int]) -> str:
    pairs = [f"{j} {int(v)}" for j, v in enumerate(action) if int(v) != 0]
    return f"JOINT {player};" + ((" " + " ".join(pairs)) if pairs else "")


def update_newgame_line(line: str, target_frame: int) -> str:
    if not line.startswith("NEWGAME 0;"):
        return line
    prefix, rest = line.split(";", 1)
    parts = rest.strip().split()
    if parts:
        parts[0] = str(max(target_frame + 100, int(parts[0]) if parts[0].isdigit() else 0))
    return prefix + ";" + " ".join(parts)


def free_generate_long(
    model: ActionGRU,
    actions: list[tuple[int, list[int]]],
    seed_frame: int,
    target_frame: int,
) -> tuple[dict[int, list[int]], dict[str, Any]]:
    source_frames = [fr for fr, _ in actions]
    true_actions = [a for _, a in actions]
    seed_idx = max(SEQ_LEN, max((i for i, fr in enumerate(source_frames) if fr <= seed_frame), default=SEQ_LEN))

    out_frames = source_frames[: seed_idx + 1]
    generated = [a[:] for a in true_actions[: seed_idx + 1]]

    # First continue through the original source frame locations, then append every STEP_FRAMES.
    future_frames = [fr for fr in source_frames[seed_idx + 1 :] if fr <= target_frame]
    fr = source_frames[-1]
    while fr + STEP_FRAMES <= target_frame:
        fr += STEP_FRAMES
        if fr not in source_frames:
            future_frames.append(fr)
    future_frames = sorted(dict.fromkeys(future_frames))

    model.eval()
    with torch.no_grad():
        for fr in future_frames:
            seq = generated[-SEQ_LEN:]
            x = torch.tensor([[one_hot_action(a) for a in seq]], dtype=torch.float32, device=DEVICE)
            logits = model(x)[0]
            pred = torch.argmax(logits, dim=-1).cpu().tolist()
            generated.append([int(v) for v in pred])
            out_frames.append(fr)

    pred_by_frame = {fr: act for fr, act in zip(out_frames, generated)}

    # Metrics only where source truth exists after the seed.
    true_by_frame = dict(actions)
    total = correct = exact = rows = 0
    for fr, pred in pred_by_frame.items():
        if fr <= source_frames[seed_idx] or fr not in true_by_frame:
            continue
        true = true_by_frame[fr]
        rows += 1
        row_ok = True
        for p, t in zip(pred, true):
            total += 1
            if int(p) == int(t):
                correct += 1
            else:
                row_ok = False
        if row_ok:
            exact += 1

    return pred_by_frame, {
        "seed_frame": seed_frame,
        "seed_idx": seed_idx,
        "seed_actual_frame": source_frames[seed_idx],
        "target_frame": target_frame,
        "generated_rows": len(pred_by_frame),
        "source_eval_rows_after_seed": rows,
        "source_eval_joint_acc": correct / max(1, total),
        "source_eval_exact_acc": exact / max(1, rows),
    }


def rewrite_template_and_append(
    pred_by_frame: dict[int, list[int]],
    out_rpl: Path,
    fightname: str,
    target_frame: int,
) -> dict[str, Any]:
    lines = TEMPLATE_RPL.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[str] = []
    replaced: set[int] = set()
    removed = inserted = 0
    template_frames: list[int] = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("FIGHTNAME 0;"):
            out.append(f"FIGHTNAME 0; {fightname}")
            i += 1
            continue
        if line.startswith("NEWGAME 0;"):
            out.append(update_newgame_line(line, target_frame))
            i += 1
            continue
        fr = parse_frame_number(line)
        if fr is not None:
            template_frames.append(fr)
            out.append(line)
            i += 1
            if fr in pred_by_frame:
                while i < len(lines):
                    nxt = lines[i]
                    if parse_frame_number(nxt) is not None:
                        break
                    if is_joint0_line(nxt):
                        removed += 1
                        i += 1
                        continue
                    out.append(nxt)
                    i += 1
                out.append(compact_joint_line(0, pred_by_frame[fr]))
                replaced.add(fr)
                inserted += 1
            continue
        out.append(line)
        i += 1

    max_template_frame = max(template_frames) if template_frames else 0
    appended = 0
    for fr in sorted(pred_by_frame):
        if fr <= max_template_frame:
            continue
        out.append("")
        out.append(f"FRAME {fr}; 0 0 0 0")
        out.append(compact_joint_line(0, pred_by_frame[fr]))
        appended += 1

    out_rpl.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {
        "template_lines": len(lines),
        "output_lines": len(out),
        "template_max_frame": max_template_frame,
        "target_frame": target_frame,
        "predicted_frames": len(pred_by_frame),
        "replaced_frames": len(replaced),
        "appended_frames": appended,
        "missing_predicted_template_frames": sorted((set(pred_by_frame) & set(template_frames)) - replaced),
        "removed_joint0_lines": removed,
        "inserted_joint0_lines": inserted,
    }


def copy_to_steam(path: Path) -> list[str]:
    copied = []
    for d in (STEAM_REPLAY_DIR, STEAM_PARKOUR_DIR):
        d.mkdir(parents=True, exist_ok=True)
        dst = d / path.name
        shutil.copy2(path, dst)
        copied.append(str(dst))
    return copied


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not TEMPLATE_RPL.exists():
        raise FileNotFoundError(TEMPLATE_RPL)

    actions = load_actions_from_dataset()
    model = load_or_train_model(actions)

    exports: list[dict[str, Any]] = []
    for run in RUNS:
        seed = int(run["seed_frame"])
        target = int(run["target_frame"])
        pred_by_frame, gen_summary = free_generate_long(model, actions, seed, target)
        name = f"xioi_assassin_action_gru_long_v41_seed{seed}_to{target}"
        out_rpl = OUT_DIR / f"{name}.rpl"
        rewrite_summary = rewrite_template_and_append(pred_by_frame, out_rpl, name, target)
        copied = copy_to_steam(out_rpl)

        actions_json = OUT_DIR / f"{name}_actions.json"
        actions_json.write_text(
            json.dumps(
                {
                    "name": name,
                    "version": 41,
                    "mode": "long_free_running_action_gru",
                    "seed_frame": seed,
                    "target_frame": target,
                    "actions": [
                        {
                            "frame": fr,
                            "values": pred_by_frame[fr],
                            "pairs": [[j, v] for j, v in enumerate(pred_by_frame[fr]) if int(v) != 0],
                        }
                        for fr in sorted(pred_by_frame)
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        info = {
            "name": name,
            "rpl": str(out_rpl),
            "actions_json": str(actions_json),
            "copied_to": copied,
            "generation": gen_summary,
            "rewrite": rewrite_summary,
        }
        exports.append(info)
        print("Generated:", out_rpl.name)
        print(json.dumps(gen_summary, indent=2))

    summary = {
        "version": 41,
        "mode": "long_free_running_action_gru",
        "model": str(MODEL_PATH),
        "dataset": str(DATASET_JSONL),
        "template": str(TEMPLATE_RPL),
        "exports": exports,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
