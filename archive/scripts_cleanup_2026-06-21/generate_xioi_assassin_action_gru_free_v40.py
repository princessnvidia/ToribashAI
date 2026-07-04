#!/usr/bin/env python3
"""
generate_xioi_assassin_action_gru_free_v40.py

V40 = free-running / autoregressive walking test.

Why a new action-GRU?
  The V38 model uses physical state features (head/chest/feet positions) as input.
  It can reproduce the replay perfectly in teacher-forced mode, but it cannot truly
  free-run without a physics simulator feeding it new POS/QAT states.

So V40 trains a tiny action-only GRU on the clean Xioi assassin walking sequence:
  previous 8 JOINT action vectors -> next JOINT action vector

Then it generates replay variants in free-running mode:
  seed with true actions up to frame 35 / 70 / 140
  then re-inject its own predicted actions until the end.

The replay export is template-safe:
  copy xioi_427_assassincreedhunter_v37.rpl
  preserve FRAME/POS/QAT/LINVEL/ANGVEL/NEWGAME/ENGAGE
  replace only JOINT 0 lines inside existing FRAME blocks.
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
REFERENCE_JSON = OUT_DIR / "xioi_assassin_reference_v38.json"
DATASET_JSONL = ROOT / "datasets/ml/xioi_assassin_walk_v38_sequences.jsonl"
TEMPLATE_RPL = OUT_DIR / "xioi_427_assassincreedhunter_v37.rpl"
MODEL_PATH = ROOT / "models/xioi_assassin_action_gru_free_v40.pt"
SUMMARY_PATH = OUT_DIR / "xioi_assassin_action_gru_free_v40_summary.json"

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
SEED_FRAMES = [35, 70, 140]
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
        for c in range(5):
            vals.append(1.0 if c == vv else 0.0)
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
            fr = row_frame(row, fallback=idx * 5)
            items[fr] = row_action(row)
    if not items:
        raise RuntimeError("No actions loaded from dataset")
    return sorted(items.items(), key=lambda x: x[0])


def build_training_pairs(actions: list[tuple[int, list[int]]]) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[list[list[float]]] = []
    ys: list[list[int]] = []
    only_actions = [a for _, a in actions]
    for i in range(SEQ_LEN, len(only_actions)):
        seq = [one_hot_action(a) for a in only_actions[i - SEQ_LEN : i]]
        xs.append(seq)
        ys.append(only_actions[i])
    if not xs:
        raise RuntimeError("Not enough actions for sequence training")
    return torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.long)


def class_weights(y: torch.Tensor) -> torch.Tensor:
    counts = Counter(int(v) for v in y.flatten().tolist())
    total = sum(counts.values())
    weights = []
    for c in range(5):
        # softened inverse frequency; prevents class 0 from dominating but avoids exploding weights
        freq = counts.get(c, 1) / max(1, total)
        weights.append((1.0 / max(freq, 1e-6)) ** 0.35)
    s = sum(weights) / len(weights)
    weights = [w / s for w in weights]
    print("Class counts:", dict(counts))
    print("Class weights:", [round(w, 4) for w in weights])
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def evaluate(model: ActionGRU, x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        logits = model(x.to(DEVICE))
        pred = torch.argmax(logits, dim=-1).cpu()
    correct = (pred == y).float()
    joint_acc = float(correct.mean().item())
    exact = float((correct.mean(dim=1) == 1.0).float().mean().item())
    nz_mask = y != 0
    if nz_mask.any():
        nonzero = float((pred[nz_mask] == y[nz_mask]).float().mean().item())
    else:
        nonzero = 0.0
    return {"joint": joint_acc, "exact": exact, "nonzero": nonzero}


def train_action_gru(actions: list[tuple[int, list[int]]]) -> tuple[ActionGRU, dict[str, Any]]:
    x, y = build_training_pairs(actions)
    model = ActionGRU().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    weights = class_weights(y)

    x_dev = x.to(DEVICE)
    y_dev = y.to(DEVICE)

    best_metric = -1.0
    best_epoch = 0
    best_state: dict[str, Any] | None = None

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
                best_epoch = epoch
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

    if best_state is not None:
        model.load_state_dict(best_state)
    summary = {"best_epoch": best_epoch, "best_metric": best_metric, "model": str(MODEL_PATH)}
    return model, summary


def free_generate(
    model: ActionGRU,
    actions: list[tuple[int, list[int]]],
    seed_frame: int,
) -> tuple[dict[int, list[int]], dict[str, Any]]:
    frames = [fr for fr, _ in actions]
    true_actions = [a for _, a in actions]

    seed_idx = max(SEQ_LEN, max((i for i, fr in enumerate(frames) if fr <= seed_frame), default=SEQ_LEN))
    generated = [a[:] for a in true_actions[: seed_idx + 1]]

    model.eval()
    with torch.no_grad():
        for i in range(seed_idx + 1, len(true_actions)):
            seq = generated[-SEQ_LEN:]
            x = torch.tensor([[one_hot_action(a) for a in seq]], dtype=torch.float32, device=DEVICE)
            logits = model(x)[0]
            pred = torch.argmax(logits, dim=-1).cpu().tolist()
            generated.append([int(v) for v in pred])

    pred_by_frame = {fr: generated[i] for i, fr in enumerate(frames)}

    # Metrics against original sequence, split after seed.
    total = correct = exact = 0
    free_total = free_correct = free_exact = 0
    pred_counts = Counter()
    true_counts = Counter()
    for i, (pred, true) in enumerate(zip(generated, true_actions)):
        row_ok = True
        row_total = 0
        row_correct = 0
        for p, t in zip(pred, true):
            pred_counts[int(p)] += 1
            true_counts[int(t)] += 1
            total += 1
            row_total += 1
            if int(p) == int(t):
                correct += 1
                row_correct += 1
            else:
                row_ok = False
        if row_ok:
            exact += 1
        if i > seed_idx:
            free_total += row_total
            free_correct += row_correct
            if row_ok:
                free_exact += 1

    summary = {
        "seed_frame": seed_frame,
        "seed_idx": seed_idx,
        "seed_actual_frame": frames[seed_idx],
        "rows": len(actions),
        "overall_joint_accuracy": correct / max(1, total),
        "overall_exact_accuracy": exact / max(1, len(actions)),
        "free_joint_accuracy": free_correct / max(1, free_total),
        "free_exact_accuracy": free_exact / max(1, len(actions) - seed_idx - 1),
        "pred_counts": pred_counts.most_common(),
        "true_counts": true_counts.most_common(),
    }
    return pred_by_frame, summary


def compact_joint_line(player: int, action: list[int]) -> str:
    pairs = [f"{j} {int(v)}" for j, v in enumerate(action) if int(v) != 0]
    return f"JOINT {player};" + ((" " + " ".join(pairs)) if pairs else "")


def parse_frame_number(line: str) -> int | None:
    m = re.match(r"^FRAME\s+(-?\d+)\s*;", line.strip())
    return int(m.group(1)) if m else None


def is_joint0_line(line: str) -> bool:
    return line.lstrip().startswith("JOINT 0;")


def rewrite_template(pred_by_frame: dict[int, list[int]], out_rpl: Path, fightname: str) -> dict[str, Any]:
    lines = TEMPLATE_RPL.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[str] = []
    replaced: set[int] = set()
    removed = inserted = 0

    i = 0
    current_frame: int | None = None
    while i < len(lines):
        line = lines[i]
        if line.startswith("FIGHTNAME 0;"):
            out.append(f"FIGHTNAME 0; {fightname}")
            i += 1
            continue
        fr = parse_frame_number(line)
        if fr is not None:
            current_frame = fr
            out.append(line)
            i += 1
            if current_frame in pred_by_frame:
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
                out.append(compact_joint_line(0, pred_by_frame[current_frame]))
                replaced.add(current_frame)
                inserted += 1
            continue
        out.append(line)
        i += 1

    out_rpl.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {
        "template_lines": len(lines),
        "output_lines": len(out),
        "replaced_frames": len(replaced),
        "missing_predicted_frames": sorted(set(pred_by_frame) - replaced),
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
    actions = load_actions_from_dataset()
    model, train_summary = train_action_gru(actions)

    all_summaries: list[dict[str, Any]] = []
    for seed in SEED_FRAMES:
        pred_by_frame, gen_summary = free_generate(model, actions, seed_frame=seed)
        name = f"xioi_assassin_action_gru_free_v40_seed{seed}"
        out_rpl = OUT_DIR / f"{name}.rpl"
        rewrite_summary = rewrite_template(pred_by_frame, out_rpl, fightname=name)
        copied = copy_to_steam(out_rpl)

        actions_json = OUT_DIR / f"{name}_actions.json"
        actions_json.write_text(
            json.dumps(
                {
                    "name": name,
                    "version": 40,
                    "mode": "action_gru_free_running",
                    "seed_frame": seed,
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

        s = {
            "name": name,
            "rpl": str(out_rpl),
            "actions_json": str(actions_json),
            "copied_to": copied,
            "generation": gen_summary,
            "rewrite": rewrite_summary,
        }
        all_summaries.append(s)
        print("Generated:", out_rpl.name)
        print(json.dumps(gen_summary, indent=2))

    summary = {
        "version": 40,
        "mode": "free_running_action_gru",
        "note": "State-GRU V38 cannot free-run without physics states; V40 trains action->action GRU and re-injects predictions.",
        "template": str(TEMPLATE_RPL),
        "dataset": str(DATASET_JSONL),
        "train": train_summary,
        "exports": all_summaries,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Summary:", SUMMARY_PATH)


if __name__ == "__main__":
    main()
