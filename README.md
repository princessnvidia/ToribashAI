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

Copy:

```
data/script/toribash_trajectory_runner_v4_5_reactive.lua
```

into your Toribash installation:

```
Toribash/data/script/
```

## 4. Start Toribash

Load the runner inside Toribash:

```
/ls toribash_trajectory_runner_v4_5_reactive.lua
```

The runner automatically handles the evaluation loop.

## 5. Launch ToribashAI

Open another terminal:

```bash
python scripts/evolution_loop_trajectory_v4_5_reactive.py
```

The Python script communicates with Toribash through the Lua runner and automatically:

- Generates candidate movements
- Evaluates trajectories
- Scores each episode
- Evolves the population
- Saves the current champion
