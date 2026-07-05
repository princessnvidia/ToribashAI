# ToribashAI

AI locomotion research using replay datasets, GRU neural networks and Lua runners to learn realistic movement in Toribash.

---

## Demo

<p align="center">
  <img src="docs/demo.gif" alt="ToribashAI Demo" width="100%">
</p>

---

## Features

- 🧠 Replay-based locomotion learning
- 🤖 GRU neural network models
- 🏃 Walking and parkour experiments
- 🎯 Automated trajectory evaluation
- ⚡ Lua runner integrated directly into Toribash
- 📊 Evolutionary optimization pipeline

---

## Tech Stack

- Python
- PyTorch
- Lua
- Toribash
- JSON
- NumPy

---

## Architecture

```
Replay Dataset
      │
      ▼
Sequence Extraction
      │
      ▼
GRU Training
      │
      ▼
Trajectory Generation
      │
      ▼
Lua Runner
      │
      ▼
Episode Evaluation
      │
      ▼
Evolution Loop
```

---

## Roadmap

- [x] Replay extraction
- [x] Dataset generation
- [x] GRU locomotion model
- [x] Automated Lua runner
- [ ] Stable walking
- [ ] Dynamic obstacle avoidance
- [ ] Parkour navigation
- [ ] Reinforcement learning experiments

---

# Quick Start

## 1. Clone the repository

```bash
git clone https://github.com/princessnvidia/ToribashAI.git
cd ToribashAI
```

## 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Install the Lua runner

Copy

```
data/script/toribash_trajectory_runner_v4_5_reactive.lua
```

into

```
Toribash/data/script/
```

---

## 4. Start Toribash

```
/ls toribash_trajectory_runner_v4_5_reactive.lua
```

---

## 5. Launch ToribashAI

```bash
python scripts/evolution_loop_trajectory_v4_5_reactive.py
```

---

## How it works

The Python process communicates with Toribash through the Lua runner and automatically:

- Generates candidate movements
- Evaluates trajectories
- Scores every episode
- Evolves the population
- Saves the current champion

---

## Status

🚧 Active Research Project
