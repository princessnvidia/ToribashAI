#!/usr/bin/env python3
from pathlib import Path
import json
import torch
import torch.nn as nn

PROJECT = Path.home() / "Documents" / "ToribashAI"

DATA_PATH = PROJECT / "datasets" / "motion_patterns" / "motion_patterns_v1.jsonl"
MODEL_PATH = PROJECT / "models" / "motion_pattern_classifier_v1.pt"

OUT_PATH = PROJECT / "datasets" / "motion_patterns" / "forward_clean_v1.jsonl"
OUT_SUMMARY = PROJECT / "datasets" / "motion_patterns" / "forward_clean_v1_summary.json"

TARGET_LABEL = "forward_y_negative"
MIN_CONFIDENCE = 0.80
MIN_DY = -1.5
MIN_Z = 4.5


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_rows():
    rows = []
    with DATA_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main():
    ckpt = torch.load(MODEL_PATH, map_location="cpu")

    feature_keys = ckpt["feature_keys"]
    label_to_id = ckpt["label_to_id"]
    id_to_label = {int(k): v for k, v in ckpt["id_to_label"].items()}

    model = MLP(ckpt["input_dim"], ckpt["output_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    mean = ckpt["mean"]
    std = ckpt["std"]

    target_id = label_to_id[TARGET_LABEL]

    rows = load_rows()

    kept = []
    rejected = 0

    with torch.no_grad():
        for row in rows:
            f = row["features"]

            x = torch.tensor([[float(f[k]) for k in feature_keys]], dtype=torch.float32)
            x = (x - mean) / std

            logits = model(x)
            probs = torch.softmax(logits, dim=1)[0]

            pred_id = int(torch.argmax(probs).item())
            pred_label = id_to_label[pred_id]
            confidence = float(probs[target_id].item())

            dy = float(f["delta_y"])
            z_min = float(f["z_min"])

            if (
                pred_label == TARGET_LABEL
                and confidence >= MIN_CONFIDENCE
                and dy <= MIN_DY
                and z_min >= MIN_Z
            ):
                row["classifier"] = {
                    "pred_label": pred_label,
                    "target_confidence": confidence,
                    "pred_confidence": float(probs[pred_id].item()),
                }
                kept.append(row)
            else:
                rejected += 1

    kept.sort(
        key=lambda r: (
            r["classifier"]["target_confidence"],
            -r["features"]["delta_y"],
            r["features"]["z_min"],
        ),
        reverse=True,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_rows": len(rows),
        "kept": len(kept),
        "rejected": rejected,
        "target_label": TARGET_LABEL,
        "min_confidence": MIN_CONFIDENCE,
        "min_dy": MIN_DY,
        "min_z": MIN_Z,
        "output": str(OUT_PATH),
        "top_20": [
            {
                "source_name": r["source_name"],
                "start_frame": r["start_frame"],
                "end_frame": r["end_frame"],
                "delta_y": r["features"]["delta_y"],
                "z_min": r["features"]["z_min"],
                "confidence": r["classifier"]["target_confidence"],
            }
            for r in kept[:20]
        ],
    }

    OUT_SUMMARY.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Filtrage terminé.")
    print(f"Input rows: {len(rows)}")
    print(f"Kept: {len(kept)}")
    print(f"Rejected: {rejected}")
    print(f"Output: {OUT_PATH}")
    print(f"Summary: {OUT_SUMMARY}")

    if kept:
        best = kept[0]
        print()
        print("BEST:")
        print(best["source_name"])
        print("frames:", best["start_frame"], "->", best["end_frame"])
        print("dy:", round(best["features"]["delta_y"], 4))
        print("z_min:", round(best["features"]["z_min"], 4))
        print("confidence:", round(best["classifier"]["target_confidence"], 4))


if __name__ == "__main__":
    main()
