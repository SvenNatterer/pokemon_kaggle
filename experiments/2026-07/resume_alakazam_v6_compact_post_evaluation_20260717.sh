#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$ROOT/logs/alakazam_v6_compact_new_pool_1m_20260717"
PYTHON="$ROOT/venv/bin/python"
TRAINED="$ROOT/models/ppo_v6_deck_bank_54_compact_a_newpool_1m_20260717.zip"
VALIDATION="$ROOT/decks/generated/opponent_factory_v6_compact_potential/validation_opponents_v6.json"
HOLDOUT="$ROOT/decks/generated/opponent_factory_v6_compact_potential/final_holdout_opponents_v6.json"

mkdir -p "$RUN_DIR"
exec >> "$RUN_DIR/pipeline.log" 2>&1
cd "$ROOT"
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1

on_error() {
  local exit_code=$?
  printf 'failed exit_code=%s at=%s stage=post_evaluation_resume\n' \
    "$exit_code" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/FAILED"
  exit "$exit_code"
}
trap on_error ERR

if [[ ! -f "$TRAINED" ]]; then
  printf 'Missing trained model: %s\n' "$TRAINED" >&2
  exit 2
fi

printf 'post_validation_started at=%s resumed=true\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/status.txt"
"$PYTHON" scripts/evaluate_submission.py \
  --candidate "$TRAINED" \
  --holdout-file "$VALIDATION" \
  --games 30 \
  --results-file "$RUN_DIR/trained_v6_compact_validation.json" \
  --progress-file "$RUN_DIR/trained_v6_compact_validation_progress.json"

printf 'post_holdout_started at=%s resumed=true\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/status.txt"
"$PYTHON" scripts/evaluate_submission.py \
  --candidate "$TRAINED" \
  --holdout-file "$HOLDOUT" \
  --games 30 \
  --results-file "$RUN_DIR/trained_v6_compact_holdout.json" \
  --progress-file "$RUN_DIR/trained_v6_compact_holdout_progress.json"

printf 'completed at=%s resumed=true\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$RUN_DIR/status.txt"
touch "$RUN_DIR/COMPLETE"
