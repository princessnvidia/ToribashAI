#!/usr/bin/env python3
"""
evolution_loop_root_walk_v1.py

Boucle d'évolution root_walk_v1:
- part de root_walk_champion_v1.json
- mute les paramètres de marche/bras/équilibre, PAS le launch hard-freeze
- exporte le candidat vers Toribash en Lua
- reset via xdotool
- lit toribashai_root_walk_result.json
- rejette les tricheurs: chute, même pied répété, perte d'alternance, mains/genoux/épaules au sol

Pré-requis:
  1. Toribash ouvert avec toribash_root_walk_runner_v1.lua chargé
  2. xdotool installé
  3. Les fichiers générés:
     python3 scripts/extract_ypska_walk_priors_v1.py
     python3 scripts/make_root_walk_seed_v1.py
     python3 scripts/export_root_walk_agent_lua_v1.py

Usage:
  cd ~/Documents/ToribashAI
  python3 scripts/evolution_loop_root_walk_v1.py --population 20

Focus:
  Le script active Toribash immédiatement au lancement, puis avant chaque reset.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

PROJECT = Path.home() / "Documents/ToribashAI"
TORIBASH_DIR = Path.home() / ".var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
SCRIPT_DIR = TORIBASH_DIR / "data/script"

CHAMPION_JSON = PROJECT / "evolution/root_walk_champion_v1.json"
SEED_JSON = PROJECT / "evolution/root_walk_seed_v1.json"
STATE_JSON = PROJECT / "evolution/root_walk_curriculum_state.json"
CANDIDATE_JSON = PROJECT / "evolution/root_walk_candidate_current.json"
AGENT_LUA = SCRIPT_DIR / "toribashai_root_walk_agent_v1.lua"
RESULT_JSON = SCRIPT_DIR / "toribashai_root_walk_result.json"
EXPORT_SCRIPT = PROJECT / "scripts/export_root_walk_agent_lua_v1.py"

POP_DIR = PROJECT / "evolution/root_walk_v1_population"
BEST_DIR = PROJECT / "evolution/root_walk_v1_best"

JOINT_VALUES = [1, 2, 3, 4]

# V1.2: autospace très agressif. Avant on envoyait 1 Space toutes les ~0.18s
# + refocus fréquent, ce qui rendait l'évolution trop lente. Ici on envoie
# une rafale rapide directement à la fenêtre Toribash.
AUTO_SPACE_INTERVAL = 0.012
AUTO_SPACE_REPEAT = 6
AUTO_SPACE_DELAY_MS = 8
FOCUS_DURING_EVAL_INTERVAL = 4.0


def load_json(path: Path, default: Any | None = None) -> Any:
    if not path.exists():
        if default is not None:
            return deepcopy(default)
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def get_path(d: dict[str, Any], path: str) -> Any:
    cur: Any = d
    for p in path.split("."):
        cur = cur[p]
    return cur


def set_path(d: dict[str, Any], path: str, value: Any) -> None:
    cur: Any = d
    parts = path.split(".")
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def normalize_agent(agent: dict[str, Any]) -> dict[str, Any]:
    agent = deepcopy(agent)
    agent.setdefault("mutation", {})
    agent.setdefault("skills", {})
    agent.setdefault("controller", {})
    agent.setdefault("arms", {})
    agent["branch"] = "root_walk_v1"
    return agent


def mutate_numeric(agent: dict[str, Any], path: str, rng: random.Random) -> bool:
    try:
        v = get_path(agent, path)
    except Exception:
        return False

    if isinstance(v, bool):
        return False

    if isinstance(v, int):
        if "frames" in path:
            nv = int(clamp(v + rng.choice([-2, -1, 1, 2]), 4, 60))
        else:
            nv = int(clamp(v + rng.choice([-1, 1]), 0, 10))
    elif isinstance(v, float):
        sigma = agent.get("mutation", {}).get("small_numeric_sigma", 0.06)
        nv = round(clamp(v + rng.gauss(0.0, sigma), 0.01, 5.0), 4)
    else:
        return False

    set_path(agent, path, nv)
    return True


def mutate_skill(agent: dict[str, Any], rng: random.Random) -> bool:
    mut = agent.get("mutation", {})
    skills = mut.get("mutable_skills", [])
    if not skills:
        return False

    skill_name = rng.choice(skills)
    skill = agent.get("skills", {}).get(skill_name)
    if not isinstance(skill, dict) or not skill:
        return False

    # On favorise les bras/jambes utiles à la marche.
    preferred = ["4", "5", "6", "7", "8", "9", "12", "13", "14", "15", "16", "17", "18", "19"]
    keys = [k for k in preferred if k in skill] or list(skill.keys())
    k = rng.choice(keys)
    old = int(skill[k])

    # Mutation locale: état voisin ou random rare.
    if rng.random() < 0.85:
        candidates = [x for x in [old - 1, old + 1] if x in JOINT_VALUES]
        if not candidates:
            candidates = JOINT_VALUES
        new = rng.choice(candidates)
    else:
        new = rng.choice(JOINT_VALUES)

    skill[k] = new
    return new != old


def mutate_agent(parent: dict[str, Any], generation: int, idx: int, rng: random.Random) -> dict[str, Any]:
    child = normalize_agent(parent)
    child["name"] = f"root_walk_v1_g{generation:05d}_c{idx:03d}"

    mut = child.get("mutation", {})
    numeric_paths = mut.get("mutable_numeric_paths", [])
    numeric_rate = float(mut.get("numeric_mutation_rate", 0.35))
    joint_rate = float(mut.get("joint_mutation_rate", 0.12))

    changes = 0

    # Quelques mutations numériques par candidat.
    for path in numeric_paths:
        if rng.random() < numeric_rate:
            if mutate_numeric(child, path, rng):
                changes += 1

    # Quelques mutations de skills.
    for _ in range(1 + rng.randint(0, 3)):
        if rng.random() < joint_rate:
            if mutate_skill(child, rng):
                changes += 1

    # Coudes: on évite de les ouvrir trop souvent dans la V1.
    for sname in ["step_left", "step_right", "push_left", "push_right", "settle_after_launch"]:
        s = child.get("skills", {}).get(sname, {})
        if isinstance(s, dict):
            if rng.random() < 0.65:
                s["6"] = int(child.get("arms", {}).get("right_elbow_closed", 2))
            if rng.random() < 0.65:
                s["9"] = int(child.get("arms", {}).get("left_elbow_closed", 2))

    child["_mutation_info"] = {"changes": changes}
    return child


def export_agent_lua(agent_json: Path) -> None:
    if EXPORT_SCRIPT.exists():
        subprocess.run(
            ["python3", str(EXPORT_SCRIPT), "--agent", str(agent_json), "--out", str(AGENT_LUA)],
            check=True,
        )
    else:
        raise FileNotFoundError(EXPORT_SCRIPT)


def set_clipboard_text(text: str) -> bool:
    # Wayland KDE peut avoir wl-copy, X11 peut avoir xclip ou xsel.
    commands = [
        ["wl-copy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
    ]
    for cmd in commands:
        try:
            p = subprocess.run(cmd, input=text.encode("utf-8"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if p.returncode == 0:
                return True
        except FileNotFoundError:
            pass
    return False


def _run_quiet(cmd: list[str], timeout: float = 1.5) -> subprocess.CompletedProcess[bytes] | None:
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def find_toribash_window() -> str | None:
    """Retourne un id de fenêtre Toribash visible, ou None.

    On essaie plusieurs recherches parce que selon Steam/Flatpak/KDE,
    le titre ou la classe de fenêtre peut varier un peu.
    """
    searches = [
        ["xdotool", "search", "--onlyvisible", "--class", "Toribash"],
        ["xdotool", "search", "--onlyvisible", "--class", "toribash"],
        ["xdotool", "search", "--onlyvisible", "--name", "Toribash"],
        ["xdotool", "search", "--onlyvisible", "--name", "toribash"],
    ]

    candidates: list[str] = []
    for cmd in searches:
        res = _run_quiet(cmd)
        if not res or res.returncode != 0:
            continue
        for line in res.stdout.decode("utf-8", "ignore").splitlines():
            wid = line.strip()
            if wid and wid not in candidates:
                candidates.append(wid)

    if not candidates:
        return None

    # On préfère une fenêtre dont le titre contient Toribash, sinon la dernière trouvée.
    for wid in reversed(candidates):
        title = _run_quiet(["xdotool", "getwindowname", wid])
        if title and b"toribash" in title.stdout.lower():
            return wid
    return candidates[-1]


def focus_toribash(max_wait: float = 3.0, announce: bool = False) -> bool:
    """Active immédiatement la fenêtre Toribash.

    Renvoie False si aucune fenêtre Toribash visible n'est trouvée.
    """
    end = time.time() + max_wait
    last_wid: str | None = None

    while time.time() < end:
        wid = find_toribash_window()
        if wid:
            last_wid = wid
            # windowactivate suffit en général sous KDE/X11. windowfocus aide parfois.
            _run_quiet(["xdotool", "windowactivate", "--sync", wid], timeout=2.0)
            _run_quiet(["xdotool", "windowfocus", "--sync", wid], timeout=2.0)
            time.sleep(0.04)

            active = _run_quiet(["xdotool", "getactivewindow"], timeout=1.0)
            if active and active.stdout.decode("utf-8", "ignore").strip() == wid:
                if announce:
                    print(f"Toribash focus OK window={wid}")
                return True

            # Même si getactivewindow ment parfois avec certains WM, on considère
            # l'activation réussie si une fenêtre a été trouvée et activée.
            if announce:
                print(f"Toribash focus requested window={wid}")
            return True

        time.sleep(0.15)

    if announce:
        print(f"WARNING: fenêtre Toribash introuvable après {max_wait:.1f}s. Last={last_wid}")
    return False


def reset_toribash() -> None:
    # Sécurité: toujours remettre Toribash au premier plan avant d'envoyer les touches.
    if not focus_toribash(max_wait=2.5):
        raise RuntimeError(
            "Fenêtre Toribash introuvable. Ouvre Toribash, charge toribash_root_walk_runner_v1.lua, "
            "puis relance le script."
        )

    # Méthode fiable déjà validée: t, Ctrl+A, Backspace, Ctrl+V, Return.
    copied = set_clipboard_text("/reset")
    subprocess.run(["xdotool", "key", "t"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.08)
    subprocess.run(["xdotool", "key", "ctrl+a"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["xdotool", "key", "BackSpace"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.04)
    if copied:
        subprocess.run(["xdotool", "key", "ctrl+v"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run(["xdotool", "type", "--clearmodifiers", "/reset"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.04)
    subprocess.run(["xdotool", "key", "Return"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def tap_space() -> None:
    # Une rafale de Space accélère Toribash sans attendre 0.18s par turn.
    # --clearmodifiers évite que Shift/Ctrl restés actifs perturbent l'entrée.
    subprocess.run(
        [
            "xdotool", "key", "--clearmodifiers",
            "--repeat", str(AUTO_SPACE_REPEAT),
            "--delay", str(AUTO_SPACE_DELAY_MS),
            "space",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_result(timeout: float) -> dict[str, Any]:
    """Attend le résultat Lua en faisant avancer Toribash.

    Toribash ne simule pas tout seul après /reset : il faut envoyer Space
    régulièrement, sinon Tori reste immobile à la frame 0 et la boucle finit en timeout.
    """
    start = time.time()
    last_space = 0.0
    last_focus = 0.0

    # Premier Space immédiat pour lancer le match.
    focus_toribash(max_wait=1.0)
    tap_space()
    last_space = time.time()

    while time.time() - start < timeout:
        if RESULT_JSON.exists():
            try:
                return load_json(RESULT_JSON)
            except Exception:
                pass

        now = time.time()

        # Garde Toribash devant sans spammer l'activation de fenêtre.
        if now - last_focus >= FOCUS_DURING_EVAL_INTERVAL:
            focus_toribash(max_wait=0.35)
            last_focus = now

        # Auto-space: fait avancer les turns pendant l'évaluation.
        if now - last_space >= AUTO_SPACE_INTERVAL:
            tap_space()
            last_space = now

        time.sleep(0.012)

    return {"score": -999999.0, "reason": "timeout", "frames": 0}


def evaluate_candidate(agent: dict[str, Any], timeout: float) -> dict[str, Any]:
    save_json(CANDIDATE_JSON, agent)
    export_agent_lua(CANDIDATE_JSON)

    if RESULT_JSON.exists():
        RESULT_JSON.unlink()

    reset_toribash()
    result = wait_result(timeout)
    result.setdefault("agent_name", agent.get("name", "unknown"))
    return result


def rejection_reason(result: dict[str, Any], state: dict[str, Any]) -> str | None:
    walk = result.get("walk", {})
    body = result.get("body", {})
    arms = result.get("arms", {})

    reason = result.get("reason", "")
    if reason == "timeout":
        return "timeout"

    if reason == "fallen":
        return "fallen"

    if int(walk.get("same_foot_repeat", 0)) > 0:
        return "same_foot"

    # V1: tolérance faible, mais pas zéro au tout début.
    if int(body.get("hand_ground_frames", 0)) > 3:
        return "hands_ground"
    if int(body.get("shoulder_ground_frames", 0)) > 0:
        return "shoulder_ground"
    if int(body.get("knee_ground_frames", 0)) > 8:
        return "knee_ground"

    champ_steps = int(state.get("champion_valid_steps", 0))
    champ_alt = int(state.get("champion_alternating_steps", 0))

    valid_steps = int(walk.get("valid_steps", 0))
    alt_steps = int(walk.get("alternating_steps", 0))

    # Une fois qu'on a acquis des pas, on ne régresse pas.
    if champ_steps >= 1 and valid_steps < champ_steps:
        return f"valid_steps<{champ_steps}"

    if champ_alt >= 1 and alt_steps < champ_alt:
        return f"alternating_steps<{champ_alt}"

    if champ_alt >= 2 and alt_steps < 2:
        return "lost_core_alternance"

    if int(walk.get("valid_steps", 0)) == 0 and float(result.get("score", -999999)) > 0:
        return "score_without_steps"

    max_pec = float(arms.get("max_pec_diff", 0.0))
    if max_pec > 1.8:
        return "pec_explosion"

    return None


def selection_score(result: dict[str, Any]) -> float:
    score = float(result.get("score", -999999.0))
    walk = result.get("walk", {})
    body = result.get("body", {})
    arms = result.get("arms", {})

    # Tri additionnel côté Python: favoriser la marche propre plutôt que le score brut.
    score += int(walk.get("valid_steps", 0)) * 300
    score += int(walk.get("alternating_steps", 0)) * 600
    score += int(walk.get("stable_frames", 0)) * 0.6
    score += int(arms.get("pec_stability_frames", 0)) * 0.4
    score -= int(body.get("hand_ground_frames", 0)) * 90
    score -= int(body.get("knee_ground_frames", 0)) * 20
    score -= int(body.get("low_hip_frames", 0)) * 10
    return score


def compact_result_line(idx: int, result: dict[str, Any], rej: str | None) -> str:
    walk = result.get("walk", {})
    body = result.get("body", {})
    arms = result.get("arms", {})
    return (
        f"c{idx:03d} sel={selection_score(result):9.1f} raw={float(result.get('score', -999999)):9.1f} "
        f"frames={int(result.get('frames', 0)):3d} "
        f"steps={int(walk.get('valid_steps', 0)):2d} alt={int(walk.get('alternating_steps', 0)):2d} "
        f"same={int(walk.get('same_foot_repeat', 0)):1d} "
        f"stable={int(walk.get('stable_frames', 0)):3d} "
        f"prog={float(walk.get('forward_progress', 0.0)):6.3f} "
        f"knee={int(body.get('knee_ground_frames', 0)):3d} hand={int(body.get('hand_ground_frames', 0)):3d} "
        f"pec={int(arms.get('pec_stability_frames', 0)):3d} "
        f"reason={result.get('reason', '?')} rej={rej or '-'}"
    )


def maybe_advance_curriculum(state: dict[str, Any], result: dict[str, Any]) -> None:
    walk = result.get("walk", {})
    valid = int(walk.get("valid_steps", 0))
    alt = int(walk.get("alternating_steps", 0))
    frames = int(result.get("frames", 0))

    stage = state.get("stage", "teacher_single_step")
    if stage == "teacher_single_step" and valid >= 1 and frames >= 250:
        state["stage"] = "alternating_walk"
    if state.get("stage") == "alternating_walk" and alt >= 2:
        state["stage"] = "fast_walk_locked_by_alternance"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--population", type=int, default=20)
    ap.add_argument("--generations", type=int, default=999999)
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--focus-wait", type=float, default=4.0, help="Temps max pour activer Toribash au lancement")
    ap.add_argument("--no-initial-focus", action="store_true", help="Ne force pas le focus Toribash au démarrage")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    if not args.no_initial_focus:
        if not focus_toribash(max_wait=args.focus_wait, announce=True):
            raise SystemExit(
                "Fenêtre Toribash introuvable. Lance Toribash puis charge /ls toribash_root_walk_runner_v1.lua."
            )

    POP_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)

    if not CHAMPION_JSON.exists():
        if not SEED_JSON.exists():
            raise SystemExit("Missing champion and seed. Lance make_root_walk_seed_v1.py")
        shutil.copyfile(SEED_JSON, CHAMPION_JSON)

    state = load_json(STATE_JSON, {
        "stage": "teacher_single_step",
        "generation": 0,
        "champion_score": -10**9,
        "champion_valid_steps": 0,
        "champion_alternating_steps": 0,
        "champion_stable_frames": 0,
        "champion_pec_stability": 0,
    })

    generation = int(state.get("generation", 0))

    for _ in range(args.generations):
        parent = load_json(CHAMPION_JSON)
        parent = normalize_agent(parent)

        print(f"\n=== root_walk_v1 generation {generation} stage={state.get('stage')} ===")
        print(
            f"champ score={state.get('champion_score')} "
            f"steps={state.get('champion_valid_steps')} alt={state.get('champion_alternating_steps')} "
            f"stable={state.get('champion_stable_frames')}"
        )

        best_candidate = None
        best_result = None
        best_sel = -math.inf

        # Candidat 0 = champion actuel, utile pour vérifier que le runner fonctionne.
        candidates = [deepcopy(parent)]
        candidates[0]["name"] = f"root_walk_v1_g{generation:05d}_c000_champion_check"
        for i in range(1, args.population):
            candidates.append(mutate_agent(parent, generation, i, rng))

        for idx, cand in enumerate(candidates):
            result = evaluate_candidate(cand, args.timeout)
            rej = rejection_reason(result, state)

            line = compact_result_line(idx, result, rej)
            print(line, flush=True)

            cand_path = POP_DIR / f"g{generation:05d}_c{idx:03d}.json"
            res_path = POP_DIR / f"g{generation:05d}_c{idx:03d}_result.json"
            save_json(cand_path, cand)
            save_json(res_path, result)

            if rej is None:
                sel = selection_score(result)
                if sel > best_sel:
                    best_sel = sel
                    best_candidate = cand
                    best_result = result

        if best_candidate is not None and best_result is not None:
            current_champ_score = float(state.get("champion_score", -10**9))
            best_raw = float(best_result.get("score", -10**9))

            # Accepter si le score de sélection progresse, ou si on gagne une vraie compétence.
            walk = best_result.get("walk", {})
            gain_skill = (
                int(walk.get("valid_steps", 0)) > int(state.get("champion_valid_steps", 0))
                or int(walk.get("alternating_steps", 0)) > int(state.get("champion_alternating_steps", 0))
            )

            if best_sel > current_champ_score or gain_skill:
                save_json(CHAMPION_JSON, best_candidate)
                save_json(BEST_DIR / f"champion_g{generation:05d}.json", best_candidate)
                save_json(BEST_DIR / f"champion_g{generation:05d}_result.json", best_result)

                state["champion_score"] = best_sel
                state["champion_valid_steps"] = int(walk.get("valid_steps", 0))
                state["champion_alternating_steps"] = int(walk.get("alternating_steps", 0))
                state["champion_stable_frames"] = int(walk.get("stable_frames", 0))
                state["champion_pec_stability"] = int(best_result.get("arms", {}).get("pec_stability_frames", 0))
                maybe_advance_curriculum(state, best_result)

                print(
                    f">>> NEW CHAMPION sel={best_sel:.1f} raw={best_raw:.1f} "
                    f"steps={state['champion_valid_steps']} alt={state['champion_alternating_steps']} "
                    f"stage={state.get('stage')}"
                )
            else:
                print("No champion replacement.")
        else:
            print("No valid candidate this generation.")

        generation += 1
        state["generation"] = generation
        save_json(STATE_JSON, state)


if __name__ == "__main__":
    main()
