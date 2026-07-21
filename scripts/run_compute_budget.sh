#!/usr/bin/env bash
set -euo pipefail

# Automated, holdout-safe compute pipeline for the bank_18 agent.
#
# Default training budget at roughly 74 steps/s:
#   stage 8 + 9:       900k steps (~3.4 h)
#   3 fine-tune arms:  300k steps each (~3.4 h total)
# Evaluation usually adds only a few minutes.

cd "$(dirname "$0")/.."

if [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

PYTHON="${PYTHON:-python}"
SOURCE_MODEL="${SOURCE_MODEL:-models/stage_snapshots/ppo_v5b_deck_bank_18_stage7_mixed_league.zip}"
CURRICULUM_MODEL="${CURRICULUM_MODEL:-models/ppo_v5b_deck_bank_18_compute.zip}"
DECK="${DECK:-decks/deck_bank/bank_18.csv}"
LEAGUE_POOL="${LEAGUE_POOL:-decks/v5_curriculum_bank18_pool.json}"
VALIDATION_FILE="${VALIDATION_FILE:-decks/validation_opponents.json}"
HOLDOUT_FILE="${HOLDOUT_FILE:-decks/holdout_opponents.json}"

NUM_ENVS="${NUM_ENVS:-8}"
N_STEPS="${N_STEPS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
FINETUNE_STEPS="${FINETUNE_STEPS:-300000}"
VALIDATION_GAMES="${VALIDATION_GAMES:-150}"
HOLDOUT_GAMES="${HOLDOUT_GAMES:-200}"
RUN_HOLDOUT="${RUN_HOLDOUT:-1}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="logs/compute_budget_${RUN_ID}"
SELECTION_FILE="${RUN_DIR}/validation_selection.json"
VALIDATION_RESULTS="${RUN_DIR}/validation_results.json"
HOLDOUT_RESULTS="${RUN_DIR}/holdout_results.json"

require_file() {
  if [ ! -f "$1" ]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

run_training_arm() {
  local name="$1"
  local epochs="$2"
  local aux_coef="$3"
  local target="models/ppo_v5b_deck_bank_18_compute_ft_${name}.zip"
  local complete_marker="${target%.zip}.complete"

  FINETUNE_MODELS+=("$target")
  if [ -f "$target" ] && [ -f "$complete_marker" ]; then
    echo "Reusing completed fine-tune arm: $target"
    return 0
  fi

  echo
  echo "Fine-tune arm: $name (${FINETUNE_STEPS} steps)"
  # No completion marker means a previous arm may have been interrupted.
  # Restart it from the frozen parent to preserve a fair comparison.
  cp "$SOURCE_MODEL" "$target"

  local cmd=(
    "$PYTHON" src/train.py
    --deck "$DECK"
    --model-name "$target"
    --continue-existing
    --opp-deck "$DECK"
    --opp-pool "$LEAGUE_POOL"
    --timesteps "$FINETUNE_STEPS"
    --policy-version v5
    --feature-variant full
    --no-card-table
    --num-envs "$NUM_ENVS"
    --n-steps "$N_STEPS"
    --batch-size "$BATCH_SIZE"
    --n-epochs "$epochs"
    --lr 3e-5
    --ent-coef 0.008
    --clip-range 0.12
    --target-kl 0.03
    --belief-actor
    --belief-dim 64
    --aux-coef "$aux_coef"
    --rotate-perspective
  )
  "${cmd[@]}"
  touch "$complete_marker"
}

for required in \
  "$SOURCE_MODEL" "$DECK" "$LEAGUE_POOL" \
  "$VALIDATION_FILE" "$HOLDOUT_FILE"; do
  require_file "$required"
done

if [ "$RUN_HOLDOUT" != "0" ] && [ "$RUN_HOLDOUT" != "1" ]; then
  echo "RUN_HOLDOUT must be 0 or 1." >&2
  exit 2
fi

mkdir -p "$RUN_DIR" models/stage_snapshots

echo "============================================================"
echo "Automated compute-budget pipeline"
echo "Run directory:       $RUN_DIR"
echo "Source model:        $SOURCE_MODEL"
echo "Fine-tune steps/arm: $FINETUNE_STEPS"
echo "Validation games:    $VALIDATION_GAMES per candidate/opponent"
echo "Holdout games:       $HOLDOUT_GAMES per opponent"
echo "============================================================"

# Continue stages 8 and 9 on a copy so the stage-7 parent stays immutable.
STAGE8_MODEL="models/stage_snapshots/ppo_v5b_deck_bank_18_compute_stage8_frozen_self_play.zip"
CURRICULUM_MARKER="${CURRICULUM_MODEL%.zip}_stages8_9.complete"
if [ -f "$STAGE8_MODEL" ] && [ -f "$CURRICULUM_MODEL" ] && [ -f "$CURRICULUM_MARKER" ]; then
  echo "Reusing existing stage 8/9 curriculum outputs."
else
  CURRICULUM_START_STAGE=8
  if [ -f "$STAGE8_MODEL" ]; then
    echo "Resuming curriculum from immutable stage-8 snapshot."
    cp "$STAGE8_MODEL" "$CURRICULUM_MODEL"
    CURRICULUM_START_STAGE=9
  else
    cp "$SOURCE_MODEL" "$CURRICULUM_MODEL"
  fi
  MODEL="$CURRICULUM_MODEL" \
    NUM_ENVS="$NUM_ENVS" \
    N_STEPS="$N_STEPS" \
    BATCH_SIZE="$BATCH_SIZE" \
    START_STAGE="$CURRICULUM_START_STAGE" \
    END_STAGE=9 \
    bash scripts/train_v5_curriculum_bank18.sh
  touch "$CURRICULUM_MARKER"
fi

require_file "$CURRICULUM_MODEL"
require_file "$STAGE8_MODEL"

# Independent ablations from the exact same frozen stage-7 parent.
declare -a FINETUNE_MODELS=()
run_training_arm "epochs2" 2 0.10
run_training_arm "aux0" 1 0

# Compare every useful checkpoint on the validation set. The holdout remains
# untouched during model selection.
declare -a CANDIDATES=(
  "$SOURCE_MODEL"
  "$STAGE8_MODEL"
  "$CURRICULUM_MODEL"
)
CANDIDATES+=("${FINETUNE_MODELS[@]}")

VALIDATION_CMD=(
  "$PYTHON" scripts/evaluate_submission.py
  --holdout-file "$VALIDATION_FILE"
  --games "$VALIDATION_GAMES"
  --results-file "$VALIDATION_RESULTS"
  --best-candidate-file "$SELECTION_FILE"
)
for model in "${CANDIDATES[@]}"; do
  VALIDATION_CMD+=(--candidate "$model")
done
"${VALIDATION_CMD[@]}"

WINNER_LABEL="$($PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["candidate"])' "$SELECTION_FILE")"
WINNER_MODEL=""
for model in "${CANDIDATES[@]}"; do
  if [ "$(basename "${model%.zip}")" = "$WINNER_LABEL" ]; then
    WINNER_MODEL="$model"
    break
  fi
done

if [ -z "$WINNER_MODEL" ]; then
  echo "Could not map validation winner '$WINNER_LABEL' to a model path." >&2
  exit 1
fi

echo
echo "Validation winner: $WINNER_MODEL"
printf '%s\n' "$WINNER_MODEL" > "${RUN_DIR}/winner_model.txt"

# Exactly one final holdout pass, after model selection is finished.
if [ "$RUN_HOLDOUT" = "1" ]; then
  "$PYTHON" scripts/evaluate_submission.py \
    --holdout-file "$HOLDOUT_FILE" \
    --games "$HOLDOUT_GAMES" \
    --results-file "$HOLDOUT_RESULTS" \
    --candidate "$WINNER_MODEL"
fi

echo
echo "============================================================"
echo "Pipeline complete"
echo "Winner:            $WINNER_MODEL"
echo "Validation result: $VALIDATION_RESULTS"
if [ "$RUN_HOLDOUT" = "1" ]; then
  echo "Holdout result:    $HOLDOUT_RESULTS"
fi
echo "============================================================"
