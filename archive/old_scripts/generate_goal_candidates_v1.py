#!/usr/bin/env python3
import json
import random
from pathlib import Path


PROJECT_DIR = Path.home() / "Documents" / "ToribashAI"

SOURCE_RPL = PROJECT_DIR / "replays_raw" / "parkour_candidate" / "3742d82a12d0_parkour11111.rpl"
GENERATED_JSON = PROJECT_DIR / "models" / "generated_replay_like_actions_v1.json"

OUTPUT_DIR = PROJECT_DIR / "models" / "goal_candidates_v1"

REPLAY_DIR = (
    Path.home()
    / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/replay/my replays"
)

NEW_MOD = "ToribashAI/toribashai_goal_flat_v1.tbm"

NUM_JOINTS = 20
PLAYER_ID = 0

NUM_CANDIDATES = 20
MUTATION_RATE = 0.12
EXTRA_ACTIVE_CHANCE = 0.08
DROP_ACTIVE_CHANCE = 0.08
MAX_FRAMES_TO_EXPORT = 500

RANDOM_SEED = 1337


def is_frame_line(line):
    return line.strip().startswith("FRAME ")


def is_joint_line(line):
    return line.strip().startswith("JOINT ")


def is_fightname_line(line):
    return line.strip().startswith("FIGHTNAME ")


def is_newgame_line(line):
    return line.strip().startswith("NEWGAME ")


def replace_mod_in_newgame(line):
    prefix, rest = line.split(";", 1)
    parts = rest.strip().split()

    for i, part in enumerate(parts):
        if part.endswith(".tbm") or part == "classic":
            parts[i] = NEW_MOD
            break

    return prefix + ";" + " ".join(parts) + "\n"


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


def load_base_actions(path):
    data = json.loads(path.read_text(encoding="utf-8"))

    actions = []

    for item in data["frames"]:
        action = item["predicted_action"]

        if len(action) != NUM_JOINTS:
            raise ValueError(f"Action invalide: {len(action)} joints")

        actions.append([int(v) for v in action])

    return actions, data.get("summary", {})


def mutate_action(action, rng):
    mutated = list(action)

    for joint_id in range(NUM_JOINTS):
        value = mutated[joint_id]

        if value != 0 and rng.random() < DROP_ACTIVE_CHANCE:
            mutated[joint_id] = 0
            continue

        if value == 0 and rng.random() < EXTRA_ACTIVE_CHANCE:
            mutated[joint_id] = rng.randint(1, 4)
            continue

        if value != 0 and rng.random() < MUTATION_RATE:
            choices = [1, 2, 3, 4]
            choices.remove(value)
            mutated[joint_id] = rng.choice(choices)

    return mutated


def mutate_actions(base_actions, candidate_id):
    rng = random.Random(RANDOM_SEED + candidate_id)

    mutated = []

    for action in base_actions:
        mutated.append(mutate_action(action, rng))

    return mutated


def export_candidate(source_lines, actions, candidate_id):
    fightname = f"ToribashAI_goal_candidate_{candidate_id:03d}"

    output_lines = []
    frame_index = -1

    replaced_frames = 0
    inserted_joint_lines = 0
    skipped_original_player_joint_lines = 0
    newgame_replaced = False

    inside_frame = False

    for line in source_lines:
        if is_fightname_line(line):
            output_lines.append(f"FIGHTNAME 0; {fightname}\n")
            continue

        if is_newgame_line(line):
            output_lines.append(replace_mod_in_newgame(line))
            newgame_replaced = True
            continue

        if is_frame_line(line):
            frame_index += 1

            if frame_index >= MAX_FRAMES_TO_EXPORT:
                break

            inside_frame = True
            output_lines.append(line)

            if frame_index < len(actions):
                action = actions[frame_index]
            else:
                action = actions[-1]

            joint_line = joint_line_for_action(action)

            if joint_line is not None:
                output_lines.append(joint_line)
                inserted_joint_lines += 1

            replaced_frames += 1
            continue

        if inside_frame and is_joint_line(line):
            stripped = line.strip()

            if stripped.startswith(f"JOINT {PLAYER_ID};") or stripped.startswith(f"JOINT {PLAYER_ID} "):
                skipped_original_player_joint_lines += 1
                continue

        output_lines.append(line)

    if not newgame_replaced:
        raise RuntimeError("Aucune ligne NEWGAME trouvée dans le template.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{fightname}.rpl"
    output_path.write_text("".join(output_lines), encoding="utf-8")

    copied_path = None

    if REPLAY_DIR.exists():
        copied_path = REPLAY_DIR / output_path.name
        copied_path.write_text("".join(output_lines), encoding="utf-8")

    return {
        "candidate_id": candidate_id,
        "fightname": fightname,
        "output_path": str(output_path),
        "copied_path": str(copied_path) if copied_path else None,
        "replaced_frames": replaced_frames,
        "inserted_joint_lines": inserted_joint_lines,
        "skipped_original_player_joint_lines": skipped_original_player_joint_lines,
    }


def main():
    print(f"Source RPL: {SOURCE_RPL}")
    print(f"Generated JSON: {GENERATED_JSON}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(f"Replay dir: {REPLAY_DIR}")
    print(f"Mod: {NEW_MOD}")
    print()

    base_actions, summary = load_base_actions(GENERATED_JSON)

    print("Base action summary:")
    print(json.dumps(summary, indent=2))
    print()

    source_lines = SOURCE_RPL.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines(keepends=True)

    manifest = {
        "config": {
            "num_candidates": NUM_CANDIDATES,
            "mutation_rate": MUTATION_RATE,
            "extra_active_chance": EXTRA_ACTIVE_CHANCE,
            "drop_active_chance": DROP_ACTIVE_CHANCE,
            "max_frames_to_export": MAX_FRAMES_TO_EXPORT,
            "random_seed": RANDOM_SEED,
            "mod": NEW_MOD,
            "source_rpl": str(SOURCE_RPL),
            "generated_json": str(GENERATED_JSON),
        },
        "candidates": [],
    }

    for candidate_id in range(NUM_CANDIDATES):
        if candidate_id == 0:
            actions = base_actions
        else:
            actions = mutate_actions(base_actions, candidate_id)

        info = export_candidate(
            source_lines=source_lines,
            actions=actions,
            candidate_id=candidate_id,
        )

        manifest["candidates"].append(info)

        print(
            f"[{candidate_id:03d}] saved {info['fightname']} "
            f"frames={info['replaced_frames']} "
            f"joint_lines={info['inserted_joint_lines']}"
        )

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Manifest saved: {manifest_path}")
    print()
    print("Dans Toribash, ouvre my replays puis teste :")
    print("ToribashAI_goal_candidate_000")
    print("ToribashAI_goal_candidate_001")
    print("...")
    print()
    print("Note celui qui avance le plus vers la cible rouge.")


if __name__ == "__main__":
    main()
