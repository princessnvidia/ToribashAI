#!/usr/bin/env python3
import copy
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
SCRIPT_DIR = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"

SEED = ROOT / "evolution/trajectory_seed_v4_3_xioi_commands.json"
CHAMPION = ROOT / "evolution/trajectory_champion_v4_3_xioi_commands.json"
STATE = ROOT / "evolution/trajectory_v4_3_curriculum_state.json"

AGENT_LUA = SCRIPT_DIR / "toribashai_agent_current.lua"
RESULT = SCRIPT_DIR / "toribashai_episode_result.json"

POP_DIR = ROOT / "evolution/population_trajectory_v4_3"
BEST_DIR = ROOT / "evolution/best_trajectory_v4_3"

POP_SIZE = 12
GENERATIONS = 999999
RESET_COMMAND = "/lm ToribashAI/toribashai_xioi_city_noobj_v1.tbm"

JOINT_VALUES = [1, 2, 3, 4]

ABS_JOINT = 0
LUMBAR_JOINT = 1
CHEST_JOINT = 2
BALANCE_JOINTS = {ABS_JOINT, LUMBAR_JOINT, CHEST_JOINT}

LEG_JOINTS = {14, 15, 16, 17, 18, 19}
CORE_JOINTS = {0, 1, 2, 3, 4, 5, 6, 7}
ARM_JOINTS = {8, 9, 10, 11, 12, 13}

INITIAL_FREEZE = 126
MAX_FRAME = 428

BASE_MIN_VALID_STEPS = 1
BASE_MIN_PEC_STABILITY = 45
PEC_MARGIN_RATIO = 0.90
MAX_PEC_DIFF = 0.70

MIN_STEPS_BEFORE_FALL_OK = 1
MIN_PEC_BEFORE_FALL_OK = 45
MAX_KNEE_ANKLE_FAIL = 0


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2))


def normalize_basic(agent):
    child = copy.deepcopy(agent)
    child["freeze_until"] = INITIAL_FREEZE
    child["loop_length"] = MAX_FRAME

    clean = []
    for cmd in sorted(child.get("commands", []), key=lambda c: int(c.get("frame", 0))):
        frame = max(0, min(MAX_FRAME, int(cmd.get("frame", 0))))
        pairs = []
        seen = set()

        for pair in cmd.get("pairs", []):
            if len(pair) != 2:
                continue
            j, v = int(pair[0]), int(pair[1])
            if 0 <= j <= 19 and v in JOINT_VALUES and j not in seen:
                pairs.append([j, v])
                seen.add(j)

        if pairs:
            clean.append({"frame": frame, "pairs": pairs})

    child["commands"] = clean
    return child


def load_seed_agent():
    return normalize_basic(load_json(SEED))


def frozen_seed_commands():
    seed = load_seed_agent()
    return [
        copy.deepcopy(cmd)
        for cmd in seed.get("commands", [])
        if int(cmd.get("frame", 0)) < INITIAL_FREEZE
    ]


def restore_frozen_launch(agent):
    child = normalize_basic(agent)

    frozen = frozen_seed_commands()
    mutable = [
        copy.deepcopy(cmd)
        for cmd in child.get("commands", [])
        if int(cmd.get("frame", 0)) >= INITIAL_FREEZE
    ]

    child["commands"] = sorted(frozen + mutable, key=lambda c: int(c["frame"]))
    child["freeze_until"] = INITIAL_FREEZE
    child["loop_length"] = MAX_FRAME
    return child


def normalize(agent):
    return restore_frozen_launch(agent)


def export_lua(agent):
    agent = normalize(agent)

    lines = [
        "TORIBASHAI_AGENT = {}",
        f'TORIBASHAI_AGENT.name = "{agent.get("name", "trajectory_v43")}"',
        "TORIBASHAI_AGENT.loop_length = 428",
        f"TORIBASHAI_AGENT.freeze_until = {INITIAL_FREEZE}",
        "TORIBASHAI_AGENT.commands = {",
    ]

    for cmd in sorted(agent["commands"], key=lambda c: int(c["frame"])):
        lines.append("  {")
        lines.append(f"    frame = {int(cmd['frame'])},")
        lines.append("    pairs = {")
        for j, v in cmd.get("pairs", []):
            lines.append(f"      {{ {int(j)}, {int(v)} }},")
        lines.append("    },")
        lines.append("  },")

    lines.append("}")
    lines.append("return TORIBASHAI_AGENT")
    AGENT_LUA.write_text("\n".join(lines), encoding="utf-8")


def toribash_command(command, after=0.25):
    subprocess.run(["xdotool", "search", "--name", "Toribash", "windowactivate", "--sync"], check=False)
    time.sleep(0.08)

    p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
    p.communicate(command.encode())

    for key, delay in [
        ("t", 0.08),
        ("ctrl+a", 0.03),
        ("BackSpace", 0.03),
        ("ctrl+v", 0.03),
        ("Return", 0.05),
    ]:
        subprocess.run(["xdotool", "key", key], check=False)
        time.sleep(delay)

    time.sleep(after)


def reset_toribash():
    toribash_command(RESET_COMMAND, after=0.75)


def load_runner_and_start():
    toribash_command("/ls toribash_trajectory_runner_v4_3.lua", after=0.35)
    subprocess.run(["xdotool", "key", "space"], check=False)
    time.sleep(0.20)


def wait_result(timeout=55):
    start = time.time()
    while time.time() - start < timeout:
        if RESULT.exists():
            try:
                r = load_json(RESULT)
                RESULT.unlink(missing_ok=True)
                return r
            except Exception:
                pass
        time.sleep(0.20)

    return {
        "score": -999999,
        "reason": "timeout",
        "valid_steps": 0,
        "pec_stability": 0,
        "max_pec_diff": 999,
    }


def mutation_rate_for_frame(frame):
    if frame < INITIAL_FREEZE:
        return 0.0
    if frame <= 220:
        return 0.035
    if frame <= 300:
        return 0.012
    return 0.004


def mutate(agent):
    child = normalize(agent)
    mutations = 0

    for cmd in child["commands"]:
        frame = int(cmd["frame"])

        if frame < INITIAL_FREEZE:
            continue

        rate = mutation_rate_for_frame(frame)

        for pair in cmd["pairs"]:
            j = int(pair[0])

            if j in BALANCE_JOINTS:
                bonus = 2.6
            elif j in LEG_JOINTS:
                bonus = 1.1
            elif j in CORE_JOINTS:
                bonus = 0.75
            elif j in ARM_JOINTS:
                bonus = 0.45
            else:
                bonus = 0.5

            if random.random() < rate * bonus:
                old = pair[1]
                pair[1] = random.choice([v for v in JOINT_VALUES if v != old])
                mutations += 1

        if random.random() < (0.055 if frame <= 220 else 0.012):
            existing = {p[0] for p in cmd["pairs"]}
            pool = list((LEG_JOINTS | CORE_JOINTS) - existing)

            for bj in BALANCE_JOINTS:
                if bj not in existing:
                    pool += [bj, bj, bj, bj]

            if pool:
                cmd["pairs"].append([random.choice(pool), random.choice(JOINT_VALUES)])
                mutations += 1

    for cmd in child["commands"]:
        frame = int(cmd["frame"])
        if frame < INITIAL_FREEZE:
            continue

        if random.random() < (0.030 if frame <= 220 else 0.006):
            cmd["frame"] = max(INITIAL_FREEZE, min(MAX_FRAME, frame + random.choice([-5, 5])))
            mutations += 1

    child["mutations"] = mutations
    child["name"] = child.get("name", "trajectory_v43") + "_mut"
    return normalize(child)


def load_state():
    if STATE.exists():
        return load_json(STATE)

    return {
        "champion_score": -999999,
        "champion_valid_steps": BASE_MIN_VALID_STEPS,
        "champion_pec_stability": BASE_MIN_PEC_STABILITY,
    }


def get_knee_ankle_fail(result):
    for key in [
        "knee_ankle_fail",
        "bad_knee_ankle_steps",
        "knee_not_above_ankle",
        "left_knee_not_above_ankle",
        "right_knee_not_above_ankle",
    ]:
        if key in result:
            try:
                return int(result.get(key, 0) or 0)
            except Exception:
                return 999
    return 0


def effective_valid_steps(result):
    valid_steps = int(result.get("valid_steps", 0) or 0)
    knee_ankle_fail = get_knee_ankle_fail(result)
    return max(0, valid_steps - knee_ankle_fail)


def has_real_walk_before_fall(result):
    valid_steps = effective_valid_steps(result)
    pec = int(result.get("pec_stability", 0) or 0)
    max_pec_diff = float(result.get("max_pec_diff", 999) or 999)
    same = int(result.get("same_foot", 0) or 0)
    knee_ankle_fail = get_knee_ankle_fail(result)

    return (
        valid_steps >= MIN_STEPS_BEFORE_FALL_OK
        and pec >= MIN_PEC_BEFORE_FALL_OK
        and max_pec_diff <= MAX_PEC_DIFF
        and same == 0
        and knee_ankle_fail <= MAX_KNEE_ANKLE_FAIL
    )


def reject_bad_candidate(result, state):
    reason = str(result.get("reason", ""))

    valid_steps = effective_valid_steps(result)
    pec = int(result.get("pec_stability", 0) or 0)
    max_pec_diff = float(result.get("max_pec_diff", 999) or 999)

    same = int(result.get("same_foot", 0) or 0)
    hop = int(result.get("hop_penalty", 0) or 0)
    knee_ankle_fail = get_knee_ankle_fail(result)
    forward = int(result.get("forward_fall", 0) or 0)

    # On garde la sélection ouverte à 1 pas pour mesurer les candidats,
    # mais le main() interdit de remplacer le champion sans effsteps>=2.
    min_steps = BASE_MIN_VALID_STEPS

    # Phase marche: tant qu'on n'a pas encore 2 pas effectifs,
    # on ne laisse pas les pecs bloquer l'exploration.
    if int(state.get("champion_valid_steps", 0)) < 2:
        min_pec = 40
    else:
        min_pec = max(BASE_MIN_PEC_STABILITY, int(state.get("champion_pec_stability", 0) * PEC_MARGIN_RATIO))

    if reason == "fallen" and not has_real_walk_before_fall(result):
        return "fallen_before_real_walk"

    if valid_steps < min_steps:
        return f"valid_steps<{min_steps}"

    if pec < min_pec:
        return f"pec_stability<{min_pec}"

    if max_pec_diff > MAX_PEC_DIFF:
        return f"max_pec_diff>{MAX_PEC_DIFF}"

    if same > 0:
        return "same_foot"

    # Phase 2: si elle trouve enfin 2 vrais pas, on accepte temporairement
    # même si le timing est encore imparfait.
    if valid_steps >= 2:
        return None

    if hop > 3:
        return "semi_pied_joint_or_hop"

    if knee_ankle_fail > MAX_KNEE_ANKLE_FAIL:
        return "knee_not_above_ankle"

    knee = int(result.get("knee_ground", 0) or 0)
    hipdrop = int(result.get("hip_drop_penalty", 0) or 0)

    # Anti "planté de pointe": elle met le pied en piquet et plonge vers l'avant.
    # On le rejette surtout en phase 1, sinon elle optimise une chute plus courte.
    if valid_steps < 2 and knee > 150 and hop >= 3:
        return "toe_stab_forward_dive"

    if valid_steps < 2 and hipdrop > 0 and hop >= 3:
        return "toe_stab_hip_drop"

    if forward > 0:
        return "forward_fall"

    return None


def evaluate(agent, gen, idx, state):
    agent = normalize(copy.deepcopy(agent))
    agent["name"] = f"trajectory_v43_g{gen:05d}_c{idx:03d}"

    path = POP_DIR / f"gen_{gen:05d}_agent_{idx:03d}.json"
    save_json(path, agent)

    export_lua(agent)
    RESULT.unlink(missing_ok=True)

    reset_toribash()
    subprocess.run(["xdotool", "key", "space"], check=False)
    time.sleep(0.20)
    result = wait_result()

    raw_score = float(result.get("score", -999999))
    score = raw_score

    result["effective_valid_steps"] = effective_valid_steps(result)
    result["knee_ankle_fail"] = get_knee_ankle_fail(result)

    reject = reject_bad_candidate(result, state)
    if reject:
        score = -999999
        result["rejected"] = True
        result["reject_reason"] = reject
    else:
        result["rejected"] = False

    result["raw_score"] = raw_score
    result["selection_score"] = score

    save_json(POP_DIR / f"gen_{gen:05d}_agent_{idx:03d}_result.json", result)
    return score, result, path


def main():
    POP_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)

    if not CHAMPION.exists():
        shutil.copy(SEED, CHAMPION)

    champion = normalize(load_json(CHAMPION))
    save_json(CHAMPION, champion)

    state = load_state()

    print("Toribash:")
    print("  /lm ToribashAI/toribashai_xioi_city_noobj_v1.tbm")
    print("  /ls toribash_trajectory_runner_v4_3.lua + Space automatique")
    print("")
    print("V4.6 explore early Python reset propre:")
    print(f"  freeze parfait frames 0-{INITIAL_FREEZE - 1}")
    print("  chute acceptée si 1 pas effectif + pec>=45 + maxpec<=0.70")
    print("  pas invalidé si genou pas au-dessus de la cheville")
    print("  mutations boostées abs/lumbar/chest")
    input("Quand c'est prêt, Entrée... ")

    print("Chargement unique du runner...")
    load_runner_and_start()

    for gen in range(GENERATIONS):
        print("\n" + "=" * 72)
        print(
            f"GEN {gen} | freeze={INITIAL_FREEZE} "
            f"| required_steps>={state.get('champion_valid_steps')} "
            f"| required_pec>={int(state.get('champion_pec_stability', 0) * PEC_MARGIN_RATIO)} "
            f"| max_pec_diff<={MAX_PEC_DIFF}"
        )
        print("=" * 72)

        candidates = [copy.deepcopy(champion)]
        candidates += [mutate(champion) for _ in range(POP_SIZE - 1)]

        best_score = -999999
        best_result = None
        best_path = None

        for i, cand in enumerate(candidates):
            score, result, path = evaluate(cand, gen, i, state)

            print(
                f"c{i:03d} select={score:.1f} raw={float(result.get('raw_score', -999999)):.1f} "
                f"steps={result.get('valid_steps')} effsteps={result.get('effective_valid_steps')} "
                f"pec={result.get('pec_stability')} "
                f"maxpec={result.get('max_pec_diff')} "
                f"same={result.get('same_foot')} hop={result.get('hop_penalty')} "
                f"knee={result.get('knee_ground')} kneeankle={result.get('knee_ankle_fail')} "
                f"core={result.get('core_recovery_bonus')} pecbonus={result.get('pec_recovery_bonus')} "
                f"hiplift={result.get('hip_lift_bonus')} hipdrop={result.get('hip_drop_penalty')} "
                f"ff={result.get('forward_fall')} "
                f"reason={result.get('reason')} "
                f"rej={result.get('reject_reason', '')}"
            )

            if score > best_score:
                best_score = score
                best_result = result
                best_path = path

        if best_result is None:
            print("Pas de candidat sélectionnable")
            continue

        best_effsteps = int(best_result.get("effective_valid_steps", 0) or 0)

        if best_effsteps < 2:
            print("Pas de nouveau champion: il faut maintenant effsteps>=2")
            continue

        if best_score > float(state.get("champion_score", -999999)):
            champion = normalize(load_json(best_path))

            state["champion_score"] = best_score
            state["champion_valid_steps"] = max(
                int(state.get("champion_valid_steps", BASE_MIN_VALID_STEPS)),
                int(best_result.get("effective_valid_steps", 0) or 0),
            )
            state["champion_pec_stability"] = max(
                int(state.get("champion_pec_stability", BASE_MIN_PEC_STABILITY)),
                int(best_result.get("pec_stability", 0) or 0),
            )

            save_json(STATE, state)
            save_json(CHAMPION, champion)

            out = BEST_DIR / (
                f"champion_gen_{gen:05d}_score_{best_score:.2f}"
                f"_steps_{state['champion_valid_steps']}"
                f"_pec_{state['champion_pec_stability']}.json"
            )
            save_json(out, champion)

            print("💜 NOUVEAU CHAMPION:", out)
            print("STATE:", state)
        else:
            print("Pas mieux")


if __name__ == "__main__":
    main()
