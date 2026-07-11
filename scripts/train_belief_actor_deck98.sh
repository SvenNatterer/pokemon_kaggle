#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source venv/bin/activate || true
export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DECK="${DECK:-decks/deck_98.csv}"
MODEL_NAME="${MODEL_NAME:-models/ppo_belief_deck_98.zip}"
OPP_MODEL="${OPP_MODEL:-models/ppo_v4_deck_98.zip}"
TIMESTEPS="${TIMESTEPS:-500000}"
NUM_ENVS="${NUM_ENVS:-8}"
BELIEF_DIM="${BELIEF_DIM:-64}"
AUX_COEF="${AUX_COEF:-0.1}"

cmd=(
  python src/train.py
  --deck "$DECK"
  --model-name "$MODEL_NAME"
  --opp-deck "$DECK"
  --timesteps "$TIMESTEPS"
  --num-envs "$NUM_ENVS"
  --checkpoint-interval 100000
  --keep-checkpoints 2
  --n-epochs 4
  --clip-range 0.2
  --batch-size 1024
  --belief-actor
  --belief-dim "$BELIEF_DIM"
  --aux-coef "$AUX_COEF"
)

if [ -f "$OPP_MODEL" ]; then
  cmd+=(--opp-model "$OPP_MODEL")
else
  echo "Warning: opponent model not found: $OPP_MODEL"
  echo "Training will use the environment fallback opponent."
fi

echo "Starting belief-actor experiment:"
printf ' %q' "${cmd[@]}"
echo

"${cmd[@]}"
