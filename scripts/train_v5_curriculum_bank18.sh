#!/usr/bin/env bash
set -euo pipefail

# Always execute relative to the repository root.
cd "$(dirname "$0")/.."

if [ -f "venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DECK="${DECK:-decks/deck_bank/bank_18.csv}"
MODEL="${MODEL:-models/ppo_v5b_deck_bank_18.zip}"
HOLDOUT_FILE="${HOLDOUT_FILE:-decks/holdout_opponents.json}"
VALIDATION_FILE="${VALIDATION_FILE:-decks/validation_opponents.json}"
LEAGUE_POOL="${LEAGUE_POOL:-decks/v5_curriculum_bank18_pool.json}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-models/frozen_opponents}"
STAGE_SNAPSHOT_DIR="${STAGE_SNAPSHOT_DIR:-models/stage_snapshots}"

NUM_ENVS="${NUM_ENVS:-8}"
N_STEPS="${N_STEPS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
N_EPOCHS="${N_EPOCHS:-1}"
BELIEF_DIM="${BELIEF_DIM:-64}"
AUX_COEF="${AUX_COEF:-0.10}"

STAGE1_STEPS="${STAGE1_STEPS:-250000}"
STAGE2_STEPS="${STAGE2_STEPS:-350000}"
STAGE3_STEPS="${STAGE3_STEPS:-650000}"
STAGE4_STEPS="${STAGE4_STEPS:-250000}"
STAGE5_STEPS="${STAGE5_STEPS:-250000}"
STAGE6_STEPS="${STAGE6_STEPS:-250000}"
STAGE7_STEPS="${STAGE7_STEPS:-1200000}"
STAGE8_STEPS="${STAGE8_STEPS:-350000}"
STAGE9_STEPS="${STAGE9_STEPS:-550000}"

START_STAGE="${START_STAGE:-1}"
END_STAGE="${END_STAGE:-9}"
RUN_HOLDOUT_EVAL="${RUN_HOLDOUT_EVAL:-0}"
EVAL_GAMES="${EVAL_GAMES:-100}"

# Default remains safe. Set to 1 or use --skip-holdout-check
# when no holdout manifest is available.
SKIP_HOLDOUT_CHECK="${SKIP_HOLDOUT_CHECK:-0}"

DRY_RUN=0

WEAK_47_MODEL="backup/backup/ppo_v4_deck_bank_47_checkpoint_1.zip"
STRONG_47_MODEL="models/ppo_v4_deck_bank_47.zip"
MODEL_19="backup/backup/ppo_v4_deck_bank_19.zip"
MODEL_37="backup/backup/ppo_v4_deck_bank_37_opp_stage2.zip"
MODEL_79="backup/backup/ppo_v4_deck_bank_79.zip"

usage() {
  cat <<'EOF'
Usage: scripts/train_v5_curriculum_bank18.sh [options]

V5 curriculum for bank_18.

By default, every selected training stage is checked against the frozen
holdout manifest. Use --skip-holdout-check only when the manifest is not
available and you have manually verified that the training opponents are
not intended holdout opponents.

Options:
  --start-stage N        Resume at this stage (default: 1).
  --end-stage N          Stop after this stage (default: 9).
  --skip-holdout-check   Run without holdout manifest validation.
  --dry-run              Validate and print commands without training.
  --final-holdout-eval   Evaluate once after all selected training stages.
  -h, --help             Show this help.

Useful overrides:
  MODEL=models/ppo_v5_deck_bank_18.zip
  NUM_ENVS=8
  STAGE1_STEPS=500000
  STAGE2_STEPS=500000
  START_STAGE=2
  END_STAGE=9
  RUN_HOLDOUT_EVAL=0
  SKIP_HOLDOUT_CHECK=1
EOF
}

require_option_value() {
  local option="$1"
  local remaining="$2"

  if [ "$remaining" -lt 2 ]; then
    echo "Missing value for ${option}" >&2
    usage
    exit 2
  fi
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --start-stage)
      require_option_value "$1" "$#"
      START_STAGE="$2"
      shift 2
      ;;
    --end-stage)
      require_option_value "$1" "$#"
      END_STAGE="$2"
      shift 2
      ;;
    --skip-holdout-check)
      SKIP_HOLDOUT_CHECK=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --final-holdout-eval)
      RUN_HOLDOUT_EVAL=1
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

validate_stage_range() {
  if ! [[ "$START_STAGE" =~ ^[0-9]+$ ]]; then
    echo "START_STAGE must be an integer: ${START_STAGE}" >&2
    exit 2
  fi

  if ! [[ "$END_STAGE" =~ ^[0-9]+$ ]]; then
    echo "END_STAGE must be an integer: ${END_STAGE}" >&2
    exit 2
  fi

  if [ "$START_STAGE" -lt 1 ] || [ "$START_STAGE" -gt 9 ]; then
    echo "START_STAGE must be between 1 and 9." >&2
    exit 2
  fi

  if [ "$END_STAGE" -lt 1 ] || [ "$END_STAGE" -gt 9 ]; then
    echo "END_STAGE must be between 1 and 9." >&2
    exit 2
  fi

  if [ "$START_STAGE" -gt "$END_STAGE" ]; then
    echo "START_STAGE cannot be greater than END_STAGE." >&2
    exit 2
  fi
}

validate_boolean_setting() {
  local value="$1"
  local name="$2"

  if [ "$value" != "0" ] && [ "$value" != "1" ]; then
    echo "${name} must be 0 or 1, received: ${value}" >&2
    exit 2
  fi
}

validate_stage_range
validate_boolean_setting "$SKIP_HOLDOUT_CHECK" "SKIP_HOLDOUT_CHECK"
validate_boolean_setting "$RUN_HOLDOUT_EVAL" "RUN_HOLDOUT_EVAL"

MODEL_ZIP="$MODEL"
[[ "$MODEL_ZIP" == *.zip ]] || MODEL_ZIP="${MODEL_ZIP}.zip"

MODEL_BASE="${MODEL_ZIP%.zip}"
MODEL_STEM="$(basename "$MODEL_BASE")"
SELF_SNAPSHOT="${SNAPSHOT_DIR}/${MODEL_STEM}_stage7_frozen.zip"

require_file() {
  local path="$1"
  local label="$2"

  if [ ! -f "$path" ]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

should_run_stage() {
  local stage="$1"

  [ "$stage" -ge "$START_STAGE" ] &&
    [ "$stage" -le "$END_STAGE" ]
}

check_safe() {
  local deck="$1"
  local model="${2:-}"
  local pool="${3:-}"

  if [ "$SKIP_HOLDOUT_CHECK" = "1" ]; then
    echo "WARNING: Holdout safety check skipped for this stage."
    return 0
  fi

  local reserved_file
  for reserved_file in "$HOLDOUT_FILE" "$VALIDATION_FILE"; do
    [ -f "$reserved_file" ] || continue
    local cmd=(
      python scripts/check_holdout_safe.py
      --holdout-file "$reserved_file"
      --deck "$deck"
    )
    [ -n "$model" ] && cmd+=(--model "$model")
    [ -n "$pool" ] && cmd+=(--pool "$pool")
    "${cmd[@]}"
  done
}

save_stage_snapshot() {
  local stage="$1"
  local label="$2"
  local destination="${STAGE_SNAPSHOT_DIR}/${MODEL_STEM}_stage${stage}_${label}.zip"
  [ "$DRY_RUN" = "0" ] || return 0
  require_file "$MODEL_ZIP" "stage model"
  mkdir -p "$STAGE_SNAPSHOT_DIR"
  if [ -f "$destination" ]; then
    echo "Keeping existing immutable stage snapshot: ${destination}"
    return 0
  fi
  cp "$MODEL_ZIP" "$destination"
  echo "Saved immutable stage snapshot: ${destination}"
}

check_not_already_training() {
  if [ "${ALLOW_CONCURRENT:-0}" = "1" ]; then
    return 0
  fi

  if pgrep -fl "src/train.py" |
    grep -F -- "$MODEL" >/dev/null 2>&1; then
    echo "A training process for ${MODEL} is already running." >&2
    echo "Set ALLOW_CONCURRENT=1 only if this is intentional." >&2
    exit 3
  fi
}

run_stage() {
  local stage="$1"
  local label="$2"
  local steps="$3"
  local opp_deck="$4"
  local opp_model="$5"
  local pool="$6"
  local lr="$7"
  local ent="$8"
  local clip="$9"
  local kl="${10}"
  local sparse="${11}"

  if ! should_run_stage "$stage"; then
    echo "Skipping stage ${stage}: ${label}"
    return 0
  fi

  require_file "$DECK" "target deck"
  require_file "$opp_deck" "opponent deck"

  if [ -n "$opp_model" ] &&
    ! {
      [ "$DRY_RUN" = "1" ] &&
        [ "$opp_model" = "$SELF_SNAPSHOT" ]
    }; then
    require_file "$opp_model" "opponent model"
  fi

  if [ -n "$pool" ]; then
    require_file "$pool" "opponent pool"
  fi

  check_safe "$opp_deck" "$opp_model" "$pool"

  local run_group_suffix

  if [ "$SKIP_HOLDOUT_CHECK" = "1" ]; then
    run_group_suffix="curriculum_unchecked"
  else
    run_group_suffix="holdout_safe_curriculum"
  fi

  export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${MODEL_STEM}_${run_group_suffix}}"
  export WANDB_NAME="${MODEL_STEM}_stage${stage}_${label}"
  export TB_LOG_NAME="$WANDB_NAME"

  local cmd=(
    python src/train.py
    --deck "$DECK"
    --model-name "$MODEL"
    --opp-deck "$opp_deck"
    --timesteps "$steps"
    --num-envs "$NUM_ENVS"
    --n-steps "$N_STEPS"
    --batch-size "$BATCH_SIZE"
    --n-epochs "$N_EPOCHS"
    --lr "$lr"
    --ent-coef "$ent"
    --clip-range "$clip"
    --target-kl "$kl"
    --belief-actor
    --belief-dim "$BELIEF_DIM"
    --aux-coef "$AUX_COEF"
    --rotate-perspective
  )

  [ -z "$opp_model" ] || cmd+=(--opp-model "$opp_model")
  [ -z "$pool" ] || cmd+=(--opp-pool "$pool")
  [ "$sparse" = "0" ] || cmd+=(--sparse-rewards)
  [ ! -f "$MODEL_ZIP" ] || cmd+=(--continue-existing)

  echo
  echo "============================================================"
  echo "Stage ${stage}: ${label} (${steps} steps)"
  echo "Opponent: ${opp_model:-${pool:-random legal policy}}"
  echo "Opponent deck: ${opp_deck}"
  echo "lr=${lr} ent=${ent} clip=${clip} target_kl=${kl}"
  echo "sparse=${sparse}"
  echo "============================================================"

  printf ' %q' "${cmd[@]}"
  echo

  if [ "$DRY_RUN" = "0" ]; then
    "${cmd[@]}"
    save_stage_snapshot "$stage" "$label"
  fi
}

create_self_snapshot() {
  if ! should_run_stage 8; then
    return 0
  fi

  if [ "$DRY_RUN" = "1" ]; then
    echo "Would create frozen self-play snapshot: ${SELF_SNAPSHOT}"
    return 0
  fi

  require_file "$MODEL_ZIP" "V5 model before self-play"

  mkdir -p "$SNAPSHOT_DIR"

  echo "Creating frozen self-play snapshot: ${SELF_SNAPSHOT}"
  cp "$MODEL_ZIP" "$SELF_SNAPSHOT"
}

if [ "$SKIP_HOLDOUT_CHECK" = "1" ]; then
  echo "============================================================"
  echo "WARNING: HOLDOUT SAFETY CHECK IS DISABLED"
  echo "Manifest will not be required or evaluated."
  echo "============================================================"
else
  require_file "$HOLDOUT_FILE" "holdout manifest"
  require_file "scripts/check_holdout_safe.py" "holdout checker"
fi

check_not_already_training

echo "============================================================"
echo "V5 bank_18 curriculum"
echo "Target model: ${MODEL}"
echo "Stages: ${START_STAGE}-${END_STAGE}"
echo "Dry run: ${DRY_RUN}"

if [ "$SKIP_HOLDOUT_CHECK" = "1" ]; then
  echo "Holdout safety: DISABLED"
else
  echo "Holdout safety: enabled"
  echo "Holdout manifest: ${HOLDOUT_FILE}"
  [ ! -f "$VALIDATION_FILE" ] || echo "Validation manifest protected: ${VALIDATION_FILE}"
fi

echo "Holdout evaluation during training: disabled"
echo "============================================================"

# 1: Learn basic legal sequences against an unpredictable but weak opponent.
run_stage 1 "random_mechanics" \
  "$STAGE1_STEPS" \
  "$DECK" \
  "" \
  "" \
  "3e-4" \
  "0.020" \
  "0.20" \
  "0.050" \
  "0"

# 2: Warm-up against the weaker bank_47 model.
run_stage 2 "abra_warmup" \
  "$STAGE2_STEPS" \
  "decks/deck_bank/bank_47.csv" \
  "$WEAK_47_MODEL" \
  "" \
  "1e-4" \
  "0.015" \
  "0.18" \
  "0.045" \
  "0"

# 3: Continue against the stronger bank_47 model.
run_stage 3 "abra_strong" \
  "$STAGE3_STEPS" \
  "decks/deck_bank/bank_47.csv" \
  "$STRONG_47_MODEL" \
  "" \
  "5e-5" \
  "0.010" \
  "0.15" \
  "0.035" \
  "0"

# 4-6: Distinct archetypes reduce matchup overfitting.
run_stage 4 "zorua_control" \
  "$STAGE4_STEPS" \
  "decks/deck_bank/bank_19.csv" \
  "$MODEL_19" \
  "" \
  "4e-5" \
  "0.010" \
  "0.15" \
  "0.035" \
  "0"

run_stage 5 "murkrow_disruption" \
  "$STAGE5_STEPS" \
  "decks/deck_bank/bank_37.csv" \
  "$MODEL_37" \
  "" \
  "4e-5" \
  "0.010" \
  "0.15" \
  "0.035" \
  "0"

run_stage 6 "slowpoke_defensive" \
  "$STAGE6_STEPS" \
  "decks/deck_bank/bank_79.csv" \
  "$MODEL_79" \
  "" \
  "4e-5" \
  "0.010" \
  "0.15" \
  "0.035" \
  "0"

# 7: Mix all learned concepts per episode.
run_stage 7 "mixed_league" \
  "$STAGE7_STEPS" \
  "$DECK" \
  "" \
  "$LEAGUE_POOL" \
  "3e-5" \
  "0.008" \
  "0.12" \
  "0.030" \
  "0"

# 8: Frozen self-play prevents the opponent from changing under the learner.
create_self_snapshot

run_stage 8 "frozen_self_play" \
  "$STAGE8_STEPS" \
  "$DECK" \
  "$SELF_SNAPSHOT" \
  "" \
  "2e-5" \
  "0.006" \
  "0.10" \
  "0.025" \
  "0"

# 9: Terminal-only consolidation against the league.
run_stage 9 "sparse_league_final" \
  "$STAGE9_STEPS" \
  "$DECK" \
  "" \
  "$LEAGUE_POOL" \
  "1e-5" \
  "0.004" \
  "0.10" \
  "0.020" \
  "1"

if [ "$RUN_HOLDOUT_EVAL" = "1" ]; then
  if [ "$DRY_RUN" = "1" ]; then
    echo "Would run one final holdout evaluation after training."
  else
    require_file "$MODEL_ZIP" "final V5 model"
    require_file \
      "scripts/evaluate_submission.py" \
      "holdout evaluation script"

    mkdir -p logs

    RESULTS="logs/final_holdout_${MODEL_STEM}_$(date +%Y%m%d_%H%M%S).json"

    echo
    echo "Training complete."
    echo "Running final evaluation-only holdout pass."
    echo "Results file: ${RESULTS}"

    python scripts/evaluate_submission.py \
      --candidate "$MODEL_ZIP" \
      --games "$EVAL_GAMES" \
      --results-file "$RESULTS"
  fi
fi

echo
echo "Curriculum complete: ${MODEL}"
