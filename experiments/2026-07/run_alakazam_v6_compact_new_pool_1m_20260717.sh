#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$ROOT/logs/alakazam_v6_compact_new_pool_1m_20260717"
PYTHON="$ROOT/venv/bin/python"
COMPACT_BASE="$ROOT/models/foundation/compact_potential/ppo_v6_deck_bank_54_compact_a.zip"
V6_BASE="$ROOT/models/foundation/ppo_v6_deck_bank_54_base_a.zip"
TRAINED="$ROOT/models/ppo_v6_deck_bank_54_compact_a_newpool_1m_20260717.zip"
VALIDATION="$ROOT/decks/generated/opponent_factory_v6_compact_potential/validation_opponents_v6.json"
HOLDOUT="$ROOT/decks/generated/opponent_factory_v6_compact_potential/final_holdout_opponents_v6.json"
TRAINING_POOL="$ROOT/decks/generated/opponent_factory_v6_compact_potential/training_pool_v6.json"

mkdir -p "$RUN_DIR"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

on_error() {
  local exit_code=$?
  printf 'failed exit_code=%s at=%s\n' "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/FAILED"
  exit "$exit_code"
}
trap on_error ERR

printf 'baseline_validation_started at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/status.txt"
"$PYTHON" scripts/evaluate_submission.py \
  --candidate "$V6_BASE" \
  --candidate "$COMPACT_BASE" \
  --holdout-file "$VALIDATION" \
  --games 30 \
  --results-file "$RUN_DIR/baseline_validation.json" \
  --best-candidate-file "$RUN_DIR/baseline_validation_selection.json" \
  --progress-file "$RUN_DIR/baseline_validation_progress.json"

printf 'v6_holdout_started at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/status.txt"
"$PYTHON" scripts/evaluate_submission.py \
  --candidate "$V6_BASE" \
  --holdout-file "$HOLDOUT" \
  --games 30 \
  --results-file "$RUN_DIR/v6_baseline_holdout.json" \
  --progress-file "$RUN_DIR/v6_baseline_holdout_progress.json"

if [[ -e "$TRAINED" ]]; then
  printf 'Refusing to overwrite existing target: %s\n' "$TRAINED" >&2
  exit 2
fi
cp "$COMPACT_BASE" "$TRAINED"

printf 'training_started at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/status.txt"
"$PYTHON" src/train.py \
  --deck decks/deck_bank/bank_54.csv \
  --model-name "$TRAINED" \
  --continue-existing \
  --opp-pool "$TRAINING_POOL" \
  --timesteps 1000000 \
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
  --seed 20260721 \
  --policy-version v6 \
  --feature-variant compact \
  --card-table \
  --reserved-opponents "$VALIDATION" \
  --reserved-opponents "$HOLDOUT"

printf 'post_validation_started at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/status.txt"
"$PYTHON" scripts/evaluate_submission.py \
  --candidate "$TRAINED" \
  --holdout-file "$VALIDATION" \
  --games 30 \
  --results-file "$RUN_DIR/trained_v6_compact_validation.json" \
  --progress-file "$RUN_DIR/trained_v6_compact_validation_progress.json"

printf 'post_holdout_started at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/status.txt"
"$PYTHON" scripts/evaluate_submission.py \
  --candidate "$TRAINED" \
  --holdout-file "$HOLDOUT" \
  --games 30 \
  --results-file "$RUN_DIR/trained_v6_compact_holdout.json" \
  --progress-file "$RUN_DIR/trained_v6_compact_holdout_progress.json"

printf 'completed at=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/status.txt"
touch "$RUN_DIR/COMPLETE"
