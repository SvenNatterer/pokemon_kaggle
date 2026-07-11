#!/bin/bash
set -e

# Observation V2: categorical card embeddings, structured in-play entities and
# a shared scorer over the encoded legal options. This must start from a fresh
# model; legacy models can still be supplied through --opp-model.

source venv/bin/activate
export PYTHONPATH="src:$PYTHONPATH"

DECK="${DECK:-decks/deck_98.csv}"
MODEL_NAME="${MODEL_NAME:-models/ppo_v5_deck_98.zip}"
OPP_DECK="${OPP_DECK:-$DECK}"
OPP_MODEL="${OPP_MODEL:-}"
NUM_ENVS="${NUM_ENVS:-8}"

ARGS=(
  --deck "$DECK"
  --model-name "$MODEL_NAME"
  --opp-deck "$OPP_DECK"
  --endless
  --num-envs "$NUM_ENVS"
  --n-epochs 4
  --clip-range 0.2
  --batch-size 1024
)

if [[ -n "$OPP_MODEL" ]]; then
  ARGS+=(--opp-model "$OPP_MODEL")
fi
if [[ -f "$MODEL_NAME" ]]; then
  ARGS+=(--continue-existing)
fi

python src/train.py "${ARGS[@]}"
