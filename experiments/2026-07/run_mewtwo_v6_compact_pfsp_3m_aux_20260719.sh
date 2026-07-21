#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$ROOT/logs/mewtwo_v6_compact_pfsp_3m_aux_20260719"
PYTHON="$ROOT/venv/bin/python"
SOURCE_MODEL="$ROOT/models/training_v6/ppo_v6_deck_bank_38_compact_scratch_pfsp_1m_20260718.zip"
TARGET_MODEL="$ROOT/models/training_v6/ppo_v6_deck_bank_38_compact_pfsp_plus3m_aux_20260719.zip"
TRAINING_POOL="$ROOT/decks/opponent_factory_v6_development_pool.json"
VAL_OPP_1="$ROOT/decks/holdout_opponents.json"
VAL_OPP_2="$ROOT/decks/validation_opponents.json"
VAL_OPP_3="$ROOT/decks/generated/opponent_factory_v6_compact_potential/validation_opponents_v6.json"
VAL_OPP_4="$ROOT/decks/generated/opponent_factory_v6_compact_potential/final_holdout_opponents_v6.json"

mkdir -p "$RUN_DIR"
exec >> "$RUN_DIR/pipeline.log" 2>&1
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export WANDB_NAME="D38_vs_Dpool_3000000_aux"
export WANDB_RUN_GROUP="mewtwo_v6_compact_pfsp_aux"

RESUME_EXISTING=false
if [[ "${1:-}" == "--resume" ]]; then
  RESUME_EXISTING=true
elif [[ $# -gt 0 ]]; then
  printf 'Unknown argument: %s\n' "$1" >&2
  exit 2
fi

if [[ ! -f "$SOURCE_MODEL" ]]; then
  printf 'blocked at=%s reason=source_model_missing source=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_MODEL" > "$RUN_DIR/status.txt"
  exit 1
fi

if [[ -e "$TARGET_MODEL" && "$RESUME_EXISTING" != true ]]; then
  printf 'blocked at=%s reason=target_model_exists target=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TARGET_MODEL" > "$RUN_DIR/status.txt"
  exit 1
fi

if [[ "$RESUME_EXISTING" == true && ! -e "$TARGET_MODEL" ]]; then
  printf 'blocked at=%s reason=resume_target_missing target=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$TARGET_MODEL" > "$RUN_DIR/status.txt"
  exit 1
fi

if [[ "$RESUME_EXISTING" != true ]]; then
  cp "$SOURCE_MODEL" "$TARGET_MODEL"
fi

printf 'training_started at=%s source=%s target=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SOURCE_MODEL" "$TARGET_MODEL" \
  > "$RUN_DIR/status.txt"

exec "$PYTHON" src/train.py \
  --deck decks/deck_bank/bank_38.csv \
  --model-name "$TARGET_MODEL" \
  --continue-existing \
  --opp-pool "$TRAINING_POOL" \
  --pfsp-lite \
  --pfsp-segment-episodes 200 \
  --pfsp-prior-games 4.0 \
  --pfsp-random-fraction 0.20 \
  --pfsp-max-probability 0.35 \
  --timesteps 3000000 \
  --num-envs 7 \
  --n-steps 2048 \
  --batch-size 1024 \
  --n-epochs 2 \
  --lr 0.0001 \
  --ent-coef 0.008 \
  --clip-range 0.12 \
  --target-kl 0.03 \
  --aux-coef 0.1 \
  --belief-actor \
  --belief-dim 64 \
  --rotate-perspective \
  --seed 20260719 \
  --policy-version v6 \
  --feature-variant compact \
  --card-table \
  --reserved-opponents "$VAL_OPP_1" \
  --reserved-opponents "$VAL_OPP_2" \
  --reserved-opponents "$VAL_OPP_3" \
  --reserved-opponents "$VAL_OPP_4"
