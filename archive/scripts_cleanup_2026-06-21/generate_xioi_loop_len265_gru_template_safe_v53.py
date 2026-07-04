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

OUT_TEACHER = GEN / "xioi_loop_len265_gru_v53_teacher_template_safe.rpl"
OUT_FREE = GEN / "xioi_loop_len265_gru_v53_free_template_safe.rpl"
ACTIONS_JSON = GEN / "xioi_loop_len265_gru_v53_actions.json"
SUMMARY_JSON = GEN / "xioi_loop_len265_gru_v53_summary.json"

STEAM = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
REPLAY_ROOT = STEAM / "replay"
REPLAY_PARKOUR = REPLAY_ROOT / "parkour"

FRAME_RE = re.compile(r"^FRAME\s+(\d+);")
JOINT0_RE = re.compile(r"^JOINT\s+0;")

ACTION_DIM = 20
CLASSES = 5


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


def predict_teacher_forced(model, rows: list[dict], device: torch.device) -> list[dict]:
    actions = []
    with torch.no_grad():
        for row in rows:
            x = torch.tensor([row["seq"]], dtype=torch.float32, device=device) / 4.0
            pred = model(x).argmax(dim=-1)[0].cpu().tolist()
            true = [int(v) for v in row["action"]]
            frame = int(row["frame"])
            pairs = [[j, int(v)] for j, v in enumerate(pred) if int(v) != 0]
            actions.append({"frame": frame, "values": pred, "pairs": pairs, "true": true})
    return actions


def predict_free(model, rows: list[dict], device: torch.device) -> list[dict]:
    # Same frame schedule as the dataset/template, but after the initial seed the GRU feeds on its own predictions.
    seq = [[int(v) for v in a] for a in rows[0]["seq"]]
    actions = []
    with torch.no_grad():
        for row in rows:
            x = torch.tensor([seq[-8:]], dtype=torch.float32, device=device) / 4.0
            pred = model(x).argmax(dim=-1)[0].cpu().tolist()
            frame = int(row["frame"])
            pairs = [[j, int(v)] for j, v in enumerate(pred) if int(v) != 0]
            actions.append({"frame": frame, "values": pred, "pairs": pairs})
            seq.append([int(v) for v in pred])
    return actions


def flat_joint_line(pairs: list[list[int]]) -> str | None:
    if not pairs:
        return None
    flat = " ".join(f"{int(j)} {int(v)}" for j, v in pairs if int(v) != 0)
    if not flat:
        return None
    return f"JOINT 0; {flat}"


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

        # Inside replaced frames, remove existing JOINT 0 and keep all physics lines exactly as-is.
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


def compare_teacher(actions: list[dict]) -> dict:
    total = 0
    correct = 0
    exact = 0
    true_counts = Counter()
    pred_counts = Counter()
    for a in actions:
        pred = [int(v) for v in a["values"]]
        true = [int(v) for v in a.get("true", [])]
        if not true:
            continue
        exact += int(pred == true)
        for p, t in zip(pred, true):
            total += 1
            correct += int(p == t)
            pred_counts[p] += 1
            true_counts[t] += 1
    return {
        "joint_accuracy": correct / max(1, total),
        "exact_accuracy": exact / max(1, len(actions)),
        "true_counts": true_counts.most_common(),
        "pred_counts": pred_counts.most_common(),
    }


def copy_to_steam(path: Path) -> None:
    for d in (REPLAY_ROOT, REPLAY_PARKOUR):
        d.mkdir(parents=True, exist_ok=True)
        dst = d / path.name
        shutil.copy2(path, dst)
        print("Copied to:", dst)


def main() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Rows:", len(rows), "frame_min:", rows[0]["frame"], "frame_max:", rows[-1]["frame"])

    model, ckpt, state_dim = load_model(device)
    teacher_actions = predict_teacher_forced(model, rows, device)
    free_actions = predict_free(model, rows, device)

    teacher_rewrite = rewrite_template_safely(
        teacher_actions,
        OUT_TEACHER,
        OUT_TEACHER.stem,
    )
    free_rewrite = rewrite_template_safely(
        free_actions,
        OUT_FREE,
        OUT_FREE.stem,
    )

    ACTIONS_JSON.write_text(
        json.dumps({"version": 53, "teacher_actions": teacher_actions, "free_actions": free_actions}, indent=2),
        encoding="utf-8",
    )

    for p in (OUT_TEACHER, OUT_FREE):
        copy_to_steam(p)

    summary = {
        "version": 53,
        "mode": "template_safe_rewrite_from_v51_champion",
        "template": str(TEMPLATE),
        "model": str(MODEL),
        "dataset": str(DATASET),
        "state_dim": state_dim,
        "teacher_rpl": str(OUT_TEACHER),
        "free_rpl": str(OUT_FREE),
        "actions_json": str(ACTIONS_JSON),
        "teacher_metrics": compare_teacher(teacher_actions),
        "teacher_rewrite": teacher_rewrite,
        "free_rewrite": free_rewrite,
        "free_pred_counts": Counter(v for a in free_actions for v in a["values"]).most_common(),
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
