#!/usr/bin/env python3
import copy, json, random, shutil, subprocess, time
from pathlib import Path

ROOT = Path.home() / "Documents" / "ToribashAI"
SCRIPT_DIR = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash/data/script"

SEED = ROOT / "evolution/trajectory_seed_v4.json"
CHAMPION = ROOT / "evolution/trajectory_champion_v4.json"
AGENT_LUA = SCRIPT_DIR / "toribashai_agent_current.lua"
RESULT = SCRIPT_DIR / "toribashai_episode_result.json"
POP_DIR = ROOT / "evolution/population_trajectory_v4"
BEST_DIR = ROOT / "evolution/best_trajectory_v4"

POP_SIZE = 12
GENERATIONS = 999999
FEATURES = 8
MUTATION_RATE = 0.08
MUTATION_SCALE = 0.35
RESET_COMMAND = "/lm ToribashAI/toribashai_xioi_city_v1.tbm"

CONTROL_JOINTS = [4,5,6,7,8,9,14,15,16,17,18,19]

def load_json(p): return json.loads(Path(p).read_text())
def save_json(p, o):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(o, indent=2))

def make_seed():
    weights = []
    for j in range(20):
        joint = []
        for state in range(4):
            joint.append([0.0] * FEATURES)
        weights.append(joint)

    # léger biais vivant : jambes/core ont des préférences différentes
    for j in CONTROL_JOINTS:
        for s in range(4):
            weights[j][s][0] = random.uniform(-0.2, 0.2)

    return {
        "name": "trajectory_servo_seed_v3",
        "mode": "servo_policy",
        "frames_per_action": 5,
        "weights": weights,
    }

def ensure_seed():
    if not SEED.exists():
        save_json(SEED, make_seed())
    if not CHAMPION.exists():
        shutil.copy(SEED, CHAMPION)

def export_lua(agent):
    lines = [
        "TORIBASHAI_AGENT = {}",
        f'TORIBASHAI_AGENT.name = "{agent["name"]}"',
        f"TORIBASHAI_AGENT.frames_per_action = {agent.get('frames_per_action', 5)}",
        "TORIBASHAI_AGENT.weights = {",
    ]

    for joint in agent["weights"]:
        lines.append("  {")
        for state_weights in joint:
            lines.append("    { " + ", ".join(f"{float(v):.6f}" for v in state_weights) + " },")
        lines.append("  },")

    lines += ["}", "return TORIBASHAI_AGENT"]
    AGENT_LUA.write_text("\n".join(lines))

def reset_toribash():
    subprocess.run(["xdotool", "search", "--name", "Toribash", "windowactivate", "--sync"], check=False)
    time.sleep(0.10)
    p = subprocess.Popen(["xclip", "-selection", "clipboard"], stdin=subprocess.PIPE)
    p.communicate(RESET_COMMAND.encode())
    for key, delay in [("t",.08), ("ctrl+a",.03), ("BackSpace",.03), ("ctrl+v",.03), ("Return",.03)]:
        subprocess.run(["xdotool", "key", key], check=False)
        time.sleep(delay)
    print("Reset envoyé:", RESET_COMMAND)

def wait_result(timeout=45):
    start = time.time()
    while time.time() - start < timeout:
        if RESULT.exists():
            try:
                r = load_json(RESULT)
                RESULT.unlink(missing_ok=True)
                return r
            except Exception:
                pass
        time.sleep(0.25)
    return {"score": -999999, "reason": "timeout", "frames": 0}

def mutate(agent):
    child = copy.deepcopy(agent)
    mutations = 0
    for j in CONTROL_JOINTS:
        for s in range(4):
            for k in range(FEATURES):
                if random.random() < MUTATION_RATE:
                    child["weights"][j][s][k] += random.gauss(0, MUTATION_SCALE)
                    mutations += 1
    child["mutations"] = mutations
    return child

def evaluate(agent, gen, idx):
    agent = copy.deepcopy(agent)
    agent["name"] = f"trajectory_v4_g{gen:05d}_c{idx:03d}"
    path = POP_DIR / f"gen_{gen:05d}_agent_{idx:03d}.json"
    save_json(path, agent)
    export_lua(agent)
    RESULT.unlink(missing_ok=True)
    reset_toribash()
    r = wait_result()
    score = float(r.get("score", -999999))
    save_json(POP_DIR / f"gen_{gen:05d}_agent_{idx:03d}_result.json", r)
    return score, r, path

def main():
    POP_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    ensure_seed()

    champion = load_json(CHAMPION)
    champion_score = -999999

    print("Dans Toribash lance :")
    print("/lm ToribashAI/toribashai_xioi_city_v1.tbm")
    print("/ls toribash_trajectory_runner_v4.lua")
    input("Quand c'est prêt, Entrée... ")

    for gen in range(GENERATIONS):
        print("\n" + "="*49)
        print("GENERATION", gen)
        print("="*49)

        candidates = [copy.deepcopy(champion)] + [mutate(champion) for _ in range(POP_SIZE - 1)]
        best_score = -999999
        best_path = None
        best_result = None

        for i, c in enumerate(candidates):
            print(f"\nGEN {gen} | CANDIDAT {i:03d}")
            score, result, path = evaluate(c, gen, i)
            print("SCORE =", score, result)

            if score > best_score:
                best_score, best_path, best_result = score, path, result

        if best_score > champion_score:
            champion_score = best_score
            champion = load_json(best_path)
            out = BEST_DIR / f"champion_gen_{gen:05d}_score_{champion_score:.2f}.json"
            save_json(out, champion)
            shutil.copy(out, CHAMPION)
            print("💜 NOUVEAU CHAMPION:", out)
        else:
            print("Pas mieux")

if __name__ == "__main__":
    main()
