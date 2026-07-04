#!/usr/bin/env python3
import json
from pathlib import Path


PROJECT_DIR = Path.home() / "Documents" / "ToribashAI"


GENERATED_JSON = PROJECT_DIR / "models" / "generated_replay_like_actions_v1.json"

REPLAY_DIR = Path.home() / ".var" / "app" / "com.valvesoftware.Steam" / ".local" / "share" / "Steam" / "steamapps" / "common" / "Toribash" / "replay" / "my replays"

SOURCE_RPL = PROJECT_DIR / "replays_raw" / "parkour_candidate" / "db7dc1937302_singleplayer 1.rpl"
OUTPUT_RPL = PROJECT_DIR / "models" / "ToribashAI_classic_template_v1.rpl"
NEW_FIGHTNAME = "ToribashAI_classic_template_v1"

NUM_JOINTS = 20
PLAYER_ID = 0


def is_frame_line(line):
    return line.strip().startswith("FRAME ")


def is_joint_line(line):
    return line.strip().startswith("JOINT ")


def is_fightname_line(line):
    return line.strip().startswith("FIGHTNAME ")


def joint_line_for_action(action):
    pairs = []

    for joint_id, value in enumerate(action):
        value = int(value)

        if value == 0:
            continue

        pairs.append(f"{joint_id} {value}")

    if not pairs:
        return None

    return f"JOINT {PLAYER_ID}; " + " ".join(pairs) + "\n"


def load_generated_actions(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    frames = data["frames"]

    actions = []

    for item in frames:
        action = item["predicted_action"]

        if len(action) != NUM_JOINTS:
            raise ValueError(f"Action invalide: {len(action)} joints")

        actions.append([int(v) for v in action])

    return actions, data.get("summary", {})


def main():
    print(f"Source RPL: {SOURCE_RPL}")
    print(f"Generated JSON: {GENERATED_JSON}")
    print(f"Output RPL: {OUTPUT_RPL}")

    actions, summary = load_generated_actions(GENERATED_JSON)

    print()
    print("Generated summary:")
    print(json.dumps(summary, indent=2))

    source_lines = SOURCE_RPL.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines(keepends=True)

    output_lines = []

    frame_index = -1
    last_action = actions[-1] if actions else [0] * NUM_JOINTS

    replaced_frames = 0
    repeated_last_action_frames = 0
    inserted_joint_lines = 0
    skipped_original_player_joint_lines = 0

    inside_replaced_frame = False

    for line in source_lines:
        if is_fightname_line(line):
            output_lines.append(f"FIGHTNAME 0; {NEW_FIGHTNAME}\n")
            continue

        if is_frame_line(line):
            frame_index += 1
            inside_replaced_frame = True

            output_lines.append(line)

            if frame_index < len(actions):
                action = actions[frame_index]
                replaced_frames += 1
            else:
                action = last_action
                repeated_last_action_frames += 1

            joint_line = joint_line_for_action(action)

            if joint_line is not None:
                output_lines.append(joint_line)
                inserted_joint_lines += 1

            continue

        if inside_replaced_frame and is_joint_line(line):
            stripped = line.strip()

            # On enlève uniquement les JOINT du joueur contrôlé.
            # On laisse les éventuels JOINT 1; du second personnage.
            if stripped.startswith(f"JOINT {PLAYER_ID};") or stripped.startswith(f"JOINT {PLAYER_ID} "):
                skipped_original_player_joint_lines += 1
                continue

        output_lines.append(line)

    OUTPUT_RPL.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_RPL.write_text("".join(output_lines), encoding="utf-8")

    print()
    print("Done.")
    print(f"Replaced frames: {replaced_frames}")
    print(f"Repeated last action frames: {repeated_last_action_frames}")
    print(f"Inserted JOINT lines: {inserted_joint_lines}")
    print(f"Skipped original JOINT player {PLAYER_ID} lines: {skipped_original_player_joint_lines}")
    print(f"Saved: {OUTPUT_RPL}")

    if REPLAY_DIR.exists():
        target = REPLAY_DIR / OUTPUT_RPL.name
        target.write_text("".join(output_lines), encoding="utf-8")
        print(f"Copied to Toribash replay dir: {target}")
    else:
        print(f"Replay dir not found: {REPLAY_DIR}")


if __name__ == "__main__":
    main()
