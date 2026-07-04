#!/usr/bin/env bash
set -e

TORIBASH_DIR="$HOME/.var/app/com.valvesoftware.Steam/.local/share/Steam/steamapps/common/Toribash"
SCRIPT_SRC="$HOME/Documents/ToribashAI/scripts/toribash_auto_eval.lua"
SCRIPT_DST="$TORIBASH_DIR/data/script/toribash_auto_eval.lua"

EVOLUTION_DIR="$HOME/Documents/ToribashAI/evolution"
DONE_FILE="$EVOLUTION_DIR/eval_done.txt"
RESULTS_FILE="$EVOLUTION_DIR/results.jsonl"

mkdir -p "$EVOLUTION_DIR"
mkdir -p "$EVOLUTION_DIR/current_generation"

echo "[ToribashAI] Cleaning old eval files..."

rm -f "$RESULTS_FILE"
rm -f "$DONE_FILE"
rm -f "$EVOLUTION_DIR/current_generation/"*.rpl 2>/dev/null || true

echo "running" > "$DONE_FILE"

echo "[ToribashAI] Copy Lua script..."
cp "$SCRIPT_SRC" "$SCRIPT_DST"

cd "$TORIBASH_DIR"

echo "[ToribashAI] Launch Toribash via Steam Flatpak..."
flatpak run com.valvesoftware.Steam steam://rungameid/248570 &

STEAM_LAUNCHER_PID=$!

echo "[ToribashAI] Waiting for Toribash window..."
sleep 20

WINDOW_ID=$(xdotool search --name "Toribash" | head -n 1 || true)

if [ -z "$WINDOW_ID" ]; then
    WINDOW_ID=$(xdotool search --class "toribash" | head -n 1 || true)
fi

if [ -z "$WINDOW_ID" ]; then
    echo "[ToribashAI] ERROR: Toribash window not found."
    echo "[ToribashAI] Is Steam already running? If not, open Steam once, then retry."
    kill "$STEAM_LAUNCHER_PID" 2>/dev/null || true
    exit 1
fi

xdotool windowactivate "$WINDOW_ID"
sleep 1
xdotool mousemove --window "$WINDOW_ID" 400 400 click 1
sleep 1

echo "[ToribashAI] Starting Lua eval..."

# Ouvre le chat/console Toribash
xdotool key Return
sleep 0.5

echo "[ToribashAI] Waiting for evaluation..."
while true; do
    if [ -f "$DONE_FILE" ] && grep -q "done" "$DONE_FILE"; then
        break
    fi
    sleep 2
done

echo "[ToribashAI] Evaluation done."

echo "[ToribashAI] Closing Toribash..."
pkill -f toribash_steam 2>/dev/null || true
pkill -f toribash_replay 2>/dev/null || true

echo "[ToribashAI] Results:"
cat "$RESULTS_FILE"
