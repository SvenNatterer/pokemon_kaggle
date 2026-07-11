#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DECK="${DECK:-decks/deck_bank/bank_18.csv}"
MODEL_NAME="${MODEL_NAME:-models/ppo_belief_deck_bank_18.zip}"

NUM_ENVS="${NUM_ENVS:-8}"
N_STEPS="${N_STEPS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
N_EPOCHS="${N_EPOCHS:-4}"
BELIEF_DIM="${BELIEF_DIM:-64}"
AUX_COEF="${AUX_COEF:-0.10}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-100000}"
KEEP_CHECKPOINTS="${KEEP_CHECKPOINTS:-2}"

STAGE1_STEPS="${STAGE1_STEPS:-250000}"
STAGE2_STEPS="${STAGE2_STEPS:-400000}"
STAGE3_STEPS="${STAGE3_STEPS:-150000}"
STAGE4_STEPS="${STAGE4_STEPS:-200000}"

STAGE1_OPP_MODEL="${STAGE1_OPP_MODEL:-models/backup/ppo_v4_deck_bank_18.zip}"
DECK7_MODEL="${DECK7_MODEL:-models/holdout/ppo_v4_deck_7.zip}"
DECK1_MODEL="${DECK1_MODEL:-models/ppo_deck_1.zip}"
BANK47_MODEL="${BANK47_MODEL:-models/ppo_v4_deck_bank_47.zip}"

DECK7_DECK="${DECK7_DECK:-decks/deck_7.csv}"
DECK1_DECK="${DECK1_DECK:-decks/deck_1.csv}"
BANK47_DECK="${BANK47_DECK:-decks/deck_bank/bank_47.csv}"

SNAPSHOT_DIR="${SNAPSHOT_DIR:-models/curriculum_snapshots}"
START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-4}"
RUN_EVAL="${RUN_EVAL:-0}"
EVAL_GAMES="${EVAL_GAMES:-30}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/train_belief_curriculum.sh [options]

Curriculum for belief-actor models. Defaults target the current bank_18 run.

Options:
  --deck PATH              Target deck csv.
  --model-name PATH        Target model zip/base path.
  --start-stage N          First stage to run (default: 1).
  --end-stage N            Last stage to run (default: 4).
  --eval                   Run reference evaluation after training.
  --dry-run                Print commands without running them.
  -h, --help               Show this help.

Useful environment overrides:
  STAGE1_STEPS=250000 STAGE2_STEPS=400000 STAGE3_STEPS=150000 STAGE4_STEPS=200000
  NUM_ENVS=8 BATCH_SIZE=1024 N_EPOCHS=4 BELIEF_DIM=64 AUX_COEF=0.10
  STAGE1_OPP_MODEL=models/backup/ppo_v4_deck_bank_18.zip
  DECK7_MODEL=models/holdout/ppo_v4_deck_7.zip
  DECK1_MODEL=models/ppo_deck_1.zip
  BANK47_MODEL=models/ppo_v4_deck_bank_47.zip
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --deck)
      DECK="$2"
      shift 2
      ;;
    --model-name)
      MODEL_NAME="$2"
      shift 2
      ;;
    --start-stage)
      START_STAGE="$2"
      shift 2
      ;;
    --end-stage)
      END_STAGE="$2"
      shift 2
      ;;
    --eval)
      RUN_EVAL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2
      ;;
  esac
done

MODEL_ZIP="$MODEL_NAME"
if [[ "$MODEL_ZIP" != *.zip ]]; then
  MODEL_ZIP="${MODEL_ZIP}.zip"
fi
MODEL_BASE="${MODEL_ZIP%.zip}"
MODEL_STEM="$(basename "$MODEL_BASE")"

require_file() {
  local path="$1"
  local label="$2"
  if [ ! -f "$path" ]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

latest_model_zip() {
  if [ -f "$MODEL_ZIP" ]; then
    echo "$MODEL_ZIP"
    return 0
  fi

  local latest=""
  latest="$(ls -t "${MODEL_BASE}"_checkpoint_*.zip 2>/dev/null | head -n 1 || true)"
  if [ -n "$latest" ]; then
    echo "$latest"
    return 0
  fi

  echo ""
}

should_run_stage() {
  local stage="$1"
  [ "$stage" -ge "$START_STAGE" ] && [ "$stage" -le "$END_STAGE" ]
}

check_not_already_training() {
  if [ "${ALLOW_CONCURRENT:-0}" = "1" ]; then
    return 0
  fi

  if pgrep -fl "src/train.py" | grep -F -- "$MODEL_NAME" >/dev/null 2>&1; then
    echo "A training process for ${MODEL_NAME} appears to be running." >&2
    echo "Stop it or wait for it to save before running this curriculum." >&2
    echo "Set ALLOW_CONCURRENT=1 only if you really know the files will not collide." >&2
    exit 3
  fi
}

run_train_stage() {
  local stage="$1"
  local label="$2"
  local steps="$3"
  local opp_deck="$4"
  local opp_model="$5"
  local lr="$6"
  local ent_coef="$7"
  local clip_range="$8"
  local target_kl="$9"
  local sparse="${10}"

  if ! should_run_stage "$stage"; then
    echo "Skipping stage ${stage}: ${label}"
    return 0
  fi

  require_file "$DECK" "target deck"
  require_file "$opp_deck" "opponent deck for stage ${stage}"
  if [ -n "$opp_model" ]; then
    require_file "$opp_model" "opponent model for stage ${stage}"
  fi

  local label_slug
  label_slug="$(echo "$label" | tr ' /' '__')"
  export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${MODEL_STEM}_curriculum}"
  export WANDB_NAME="${MODEL_STEM}_stage${stage}_${label_slug}"
  export TB_LOG_NAME="${MODEL_STEM}_stage${stage}_${label_slug}"

  local cmd=(
    python src/train.py
    --deck "$DECK"
    --model-name "$MODEL_NAME"
    --opp-deck "$opp_deck"
    --timesteps "$steps"
    --num-envs "$NUM_ENVS"
    --n-steps "$N_STEPS"
    --batch-size "$BATCH_SIZE"
    --n-epochs "$N_EPOCHS"
    --checkpoint-interval "$CHECKPOINT_INTERVAL"
    --keep-checkpoints "$KEEP_CHECKPOINTS"
    --lr "$lr"
    --ent-coef "$ent_coef"
    --clip-range "$clip_range"
    --target-kl "$target_kl"
    --belief-actor
    --belief-dim "$BELIEF_DIM"
    --aux-coef "$AUX_COEF"
  )

  if [ -n "$opp_model" ]; then
    cmd+=(--opp-model "$opp_model")
  fi
  if [ "$sparse" = "1" ]; then
    cmd+=(--sparse-rewards)
  fi

  echo ""
  echo "================================================="
  echo "Stage ${stage}: ${label}"
  echo "steps=${steps} lr=${lr} ent=${ent_coef} clip=${clip_range} target_kl=${target_kl} sparse=${sparse}"
  echo "opponent_deck=${opp_deck}"
  echo "opponent_model=${opp_model:-environment fallback}"
  echo "================================================="
  printf ' %q' "${cmd[@]}"
  echo

  if [ "$DRY_RUN" = "0" ]; then
    "${cmd[@]}"
  fi
}

snapshot_current_model() {
  local stage="$1"
  local label="$2"
  mkdir -p "$SNAPSHOT_DIR"

  local current_zip
  current_zip="$(latest_model_zip)"
  if [ -z "$current_zip" ]; then
    echo "No existing model or checkpoint found for ${MODEL_ZIP}; cannot create self-play snapshot." >&2
    exit 1
  fi

  local snapshot="${SNAPSHOT_DIR}/${MODEL_STEM}_${label}.zip"
  echo "Creating frozen self-play snapshot: ${snapshot}" >&2
  printf ' cp %q %q\n' "$current_zip" "$snapshot" >&2
  if [ "$DRY_RUN" = "0" ]; then
    cp "$current_zip" "$snapshot"
  fi
  echo "$snapshot"
}

check_not_already_training

echo "================================================="
echo "Belief-actor curriculum"
echo "Target deck:  ${DECK}"
echo "Target model: ${MODEL_NAME}"
echo "Stages:       ${START_STAGE}-${END_STAGE}"
echo "================================================="

# Stage 1 keeps the task familiar and uses shaped rewards for action basics.
run_train_stage 1 "same_deck_teacher" "$STAGE1_STEPS" "$DECK" "$STAGE1_OPP_MODEL" \
  "8e-5" "0.010" "0.18" "0.040" "0"

# Stage 2 freezes the current policy and teaches the model to beat its own latest style.
if should_run_stage 2; then
  SELF_STAGE2_MODEL="$(snapshot_current_model 2 "self_stage2")"
else
  SELF_STAGE2_MODEL=""
fi
run_train_stage 2 "frozen_self_play" "$STAGE2_STEPS" "$DECK" "$SELF_STAGE2_MODEL" \
  "5e-5" "0.006" "0.15" "0.035" "0"

# Stage 3 is short, sparse, and reference-based: broaden without memorizing one ladder bot.
run_train_stage 3 "kaggle_anchor_deck7" "$STAGE3_STEPS" "$DECK7_DECK" "$DECK7_MODEL" \
  "3e-5" "0.004" "0.12" "0.030" "1"
run_train_stage 3 "kaggle_anchor_deck1" "$STAGE3_STEPS" "$DECK1_DECK" "$DECK1_MODEL" \
  "3e-5" "0.004" "0.12" "0.030" "1"
run_train_stage 3 "kaggle_anchor_bank47" "$STAGE3_STEPS" "$BANK47_DECK" "$BANK47_MODEL" \
  "3e-5" "0.004" "0.12" "0.030" "1"

# Stage 4 re-centers after the reference opponents with a very conservative sparse self-play block.
if should_run_stage 4; then
  SELF_STAGE4_MODEL="$(snapshot_current_model 4 "self_stage4")"
else
  SELF_STAGE4_MODEL=""
fi
run_train_stage 4 "final_sparse_self_play" "$STAGE4_STEPS" "$DECK" "$SELF_STAGE4_MODEL" \
  "2e-5" "0.002" "0.10" "0.020" "1"

if [ "$RUN_EVAL" = "1" ]; then
  echo ""
  echo "Running reference evaluation..."
  python3 scripts/evaluate_overnight_references.py \
    --candidate "$MODEL_ZIP" \
    --games "$EVAL_GAMES"
fi

echo ""
echo "Curriculum finished for ${MODEL_NAME}"
