#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path.home() / "Documents/ToribashAI"
GEN = ROOT / "generated_replays"
DATASET = ROOT / "datasets/ml/xioi_loop_len265_v52_sequences.jsonl"
MODEL = ROOT / "models/xioi_loop_len265_gru_v52.pt"
TEMPLATE = GEN / "xioi_loop_len265_champion_v51.rpl"

STEAM = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
REPLAY_ROOT = STEAM / "replay"
REPLAY_PARKOUR = REPLAY_ROOT / "parkour"

FRAME_RE = re.compile(r"^FRAME\s+(\d+);")
JOINT0_RE = re.compile(r"^JOINT\s+0;")
ACTION_DIM = 20
CLASSES = 5
SEEDS = [8, 24, 48, 96, 160]


class GRUAction(nn.Module):
    def __init__(self, state_dim: int, hidden: int, layers: int):
        super().__init__()
        self.gru = nn.GRU(
            state_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=0.10 if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, ACTION_DIM * CLASSES),
        )

    def forward(self, x):
        z, _ = self.gru(x)
        return self.head(z[:, -1]).view(-1, ACTION_DIM, CLASSES)


def load_rows() -> list[dict]:
    if not DATASET.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET}\nRun build_xioi_loop_len265_dataset_v52.py first.")
    rows = [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda r: int(r["frame"]))
    if not rows:
        raise RuntimeError(f"Empty dataset: {DATASET}")
    return rows


def load_model(device: torch.device):
    if not MODEL.exists():
        raise FileNotFoundError(f"Missing model: {MODEL}\nRun train_xioi_loop_len265_gru_v52.py first.")
    ckpt = torch.load(MODEL, map_location=device)
    state_dim = int(ckpt.get("state_dim", 20))
    hidden = int(ckpt.get("hidden", 192))
    layers = int(ckpt.get("layers", 2))
    model = GRUAction(state_dim, hidden, layers).to(device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded model: {MODEL} hidden={hidden} layers={layers} epoch={ckpt.get('epoch') if isinstance(ckpt, dict) else '?'}")
    return model, ckpt, state_dim


def true_actions(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        values = [int(v) for v in row["action"]]
        pairs = [[j, v] for j, v in enumerate(values) if v != 0]
        out.append({"frame": int(row["frame"]), "values": values, "pairs": pairs, "mode": "true"})
    return out


def predict_free_after_seed(model, rows: list[dict], seed_rows: int, device: torch.device) -> list[dict]:
    # The first seed_rows actions are copied from the reference. After that, the model feeds on its own predictions.
    seed_rows = max(1, min(seed_rows, len(rows)))
    seq = [[int(v) for v in a] for a in rows[0]["seq"]]
    actions = []

    with torch.no_grad():
        for i, row in enumerate(rows):
            frame = int(row["frame"])
            if i < seed_rows:
                values = [int(v) for v in row["action"]]
                mode = "seed_true"
            else:
                x = torch.tensor([seq[-8:]], dtype=torch.float32, device=device) / 4.0
                values = model(x).argmax(dim=-1)[0].cpu().tolist()
                values = [int(v) for v in values]
                mode = "free_pred"

            pairs = [[j, v] for j, v in enumerate(values) if v != 0]
            actions.append({"frame": frame, "values": values, "pairs": pairs, "mode": mode})
            seq.append(values)

    return actions


def flat_joint_line(pairs: list[list[int]]) -> str | None:
    flat_pairs = [(int(j), int(v)) for j, v in pairs if int(v) != 0]
    if not flat_pairs:
        return None
    return "JOINT 0; " + " ".join(f"{j} {v}" for j, v in flat_pairs)


def rewrite_template_safely(actions: list[dict], out_path: Path, fightname: str) -> dict:
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Missing template: {TEMPLATE}\nRun generate_xioi_loop_len265_champion_v51.py first.")

    action_by_frame = {int(a["frame"]): a for a in actions}
    original = TEMPLATE.read_text(encoding="utf-8", errors="ignore").splitlines()
    out: list[str] = []

    current_frame: int | None = None
    in_replaced_frame = False
    replaced = 0
    removed_joint0 = 0
    inserted_joint0 = 0
    seen_frames = set()

    for line in original:
        m = FRAME_RE.match(line)
        if m:
            current_frame = int(m.group(1))
            seen_frames.add(current_frame)
            in_replaced_frame = current_frame in action_by_frame
            out.append(line)
            if in_replaced_frame:
                jline = flat_joint_line(action_by_frame[current_frame].get("pairs", []))
                if jline is not None:
                    out.append(jline)
                    inserted_joint0 += 1
                replaced += 1
            continue

        if line.startswith("FIGHTNAME 0;"):
            out.append(f"FIGHTNAME 0; {fightname}")
            continue

        if in_replaced_frame and JOINT0_RE.match(line):
            removed_joint0 += 1
            continue

        out.append(line)

    missing = sorted(set(action_by_frame) - seen_frames)
    out_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {
        "template_lines": len(original),
        "output_lines": len(out),
        "predicted_frames": len(action_by_frame),
        "replaced_frames": replaced,
        "missing_predicted_frames": missing,
        "removed_joint0_lines": removed_joint0,
        "inserted_joint0_lines": inserted_joint0,
    }


def copy_to_steam(path: Path) -> None:
    for d in (REPLAY_ROOT, REPLAY_PARKOUR):
        d.mkdir(parents=True, exist_ok=True)
        dst = d / path.name
        shutil.copy2(path, dst)
        print("Copied to:", dst)


def count_values(actions: list[dict]) -> list[tuple[int, int]]:
    return Counter(v for a in actions for v in a["values"]).most_common()


def compare_to_true(actions: list[dict], rows: list[dict]) -> dict:
    true_by_frame = {int(r["frame"]): [int(v) for v in r["action"]] for r in rows}
    total = correct = exact = compared = 0
    for a in actions:
        frame = int(a["frame"])
        if frame not in true_by_frame:
            continue
        compared += 1
        pred = [int(v) for v in a["values"]]
        true = true_by_frame[frame]
        exact += int(pred == true)
        for p, t in zip(pred, true):
            total += 1
            correct += int(p == t)
    return {
        "compared_frames": compared,
        "joint_accuracy_vs_reference": correct / max(1, total),
        "exact_accuracy_vs_reference": exact / max(1, compared),
    }


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Rows:", len(rows), "frame_min:", rows[0]["frame"], "frame_max:", rows[-1]["frame"])

    model, ckpt, state_dim = load_model(device)

    outputs = []

    teacher = true_actions(rows)
    teacher_path = GEN / "xioi_loop_len265_gru_long_v54_teacher_template_safe.rpl"
    teacher_rewrite = rewrite_template_safely(teacher, teacher_path, teacher_path.stem)
    copy_to_steam(teacher_path)
    outputs.append({
        "name": teacher_path.stem,
        "path": str(teacher_path),
        "mode": "reference_teacher",
        "rewrite": teacher_rewrite,
        "pred_counts": count_values(teacher),
        "metrics": compare_to_true(teacher, rows),
    })

    for seed in SEEDS:
        actions = predict_free_after_seed(model, rows, seed, device)
        out_path = GEN / f"xioi_loop_len265_gru_long_v54_seed{seed:03d}_template_safe.rpl"
        rewrite = rewrite_template_safely(actions, out_path, out_path.stem)
        copy_to_steam(out_path)
        outputs.append({
            "name": out_path.stem,
            "path": str(out_path),
            "mode": "free_after_seed",
            "seed_rows": seed,
            "rewrite": rewrite,
            "pred_counts": count_values(actions),
            "metrics": compare_to_true(actions, rows),
        })

    summary = {
        "version": 54,
        "purpose": "long template-safe robustness test over V51 len265 loop champion",
        "template": str(TEMPLATE),
        "model": str(MODEL),
        "dataset": str(DATASET),
        "state_dim": state_dim,
        "rows": len(rows),
        "seeds": SEEDS,
        "outputs": outputs,
    }
    summary_path = GEN / "xioi_loop_len265_gru_long_v54_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
