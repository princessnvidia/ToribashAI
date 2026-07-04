#!/usr/bin/env python3
import json
from pathlib import Path


PROJECT_DIR = Path.home() / "Documents" / "ToribashAI"

SOURCE_RPL = PROJECT_DIR / "replays_raw" / "parkour_candidate" / "3742d82a12d0_parkour11111.rpl"
GENERATED_JSON = PROJECT_DIR / "models" / "generated_replay_like_actions_v1.json"

OUTPUT_RPL = PROJECT_DIR / "models" / "ToribashAI_generated_v1.rpl"

NUM_JOINTS = 20


def is_frame_line(line):
    return line.strip().startswith("FRAME ")


def is_joint_line(line):
    return line.strip().startswith("JOINT ")


def parse_frame_number(line):
    parts = line.strip().split()

    if len(parts) >= 2 and parts[0] == "FRAME":
        try:
            return int(parts[1])
        except ValueError:
            return None

    return None


def joint_lines_for_action(action):
    pairs = []

    for joint_id, value in enumerate(action):
        value = int(value)

        if value == 0:
            continue

        pairs.append(f"{joint_id} {value}")

    if not pairs:
        return []

    return [
        "JOINT 0; " + " ".join(pairs) + "\n"
    ]


def load_generated_actions(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    frames = data["frames"]

    actions_by_index = {}

    for item in frames:
        frame_index = int(item["frame_index"])
        action = item["predicted_action"]

        if len(action) != NUM_JOINTS:
            raise ValueError(
                f"Action invalide frame {frame_index}: {len(action)} joints"
            )

        actions_by_index[frame_index] = [int(v) for v in action]

    return actions_by_index, data.get("summary", {})


def main():
    print(f"Source RPL: {SOURCE_RPL}")
    print(f"Generated JSON: {GENERATED_JSON}")
    print(f"Output RPL: {OUTPUT_RPL}")

    actions_by_index, summary = load_generated_actions(GENERATED_JSON)

    print()
    print("Generated summary:")
    print(json.dumps(summary, indent=2))

    source_lines = SOURCE_RPL.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines(keepends=True)

    output_lines = []

    current_generated_frame_index = -1
    inside_replaced_frame = False

    replaced_frames = 0
    inserted_joint_lines = 0
    skipped_original_joint_lines = 0

    for line in source_lines:
        if is_frame_line(line):
            current_generated_frame_index += 1
            inside_replaced_frame = current_generated_frame_index in actions_by_index

            output_lines.append(line)

            if inside_replaced_frame:
                action = actions_by_index[current_generated_frame_index]
                new_joint_lines = joint_lines_for_action(action)

                output_lines.extend(new_joint_lines)

                replaced_frames += 1
                inserted_joint_lines += len(new_joint_lines)

            continue

        if inside_replaced_frame and is_joint_line(line):
            skipped_original_joint_lines += 1
            continue

        output_lines.append(line)

    OUTPUT_RPL.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_RPL.write_text(
        "".join(output_lines),
        encoding="utf-8",
    )

    print()
    print("Done.")
    print(f"Replaced frames: {replaced_frames}")
    print(f"Inserted JOINT lines: {inserted_joint_lines}")
    print(f"Skipped original JOINT lines: {skipped_original_joint_lines}")
    print(f"Saved: {OUTPUT_RPL}")

    print()
    print("Tu peux tenter de copier ce fichier dans ton dossier Toribash/replay puis l'ouvrir :")
    print(OUTPUT_RPL)


if __name__ == "__main__":
    main()
