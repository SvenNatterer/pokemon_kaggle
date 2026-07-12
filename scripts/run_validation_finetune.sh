#!/usr/bin/env bash
# Controlled fine-tuning from one frozen parent model.  Each arm changes one
# training factor only; do not compare arms that used different parent models.
set -euo pipefail

cd "$(dirname "$0")/.."
[ ! -f venv/bin/activate ] || source venv/bin/activate
export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

BASE_MODEL="${BASE_MODEL:-models/ppo_v5_deck_bank_18_checkpoint_1.zip}"
DECK="${DECK:-decks/deck_bank/bank_18.csv}"
LEAGUE_POOL="${LEAGUE_POOL:-decks/v5_curriculum_bank18_pool.json}"
VALIDATION_FILE="${VALIDATION_FILE:-decks/validation_opponents.json}"
STEPS="${STEPS:-500000}"
NUM_ENVS="${NUM_ENVS:-8}"
N_STEPS="${N_STEPS:-2048}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
LR="${LR:-3e-5}"
ENT_COEF="${ENT_COEF:-0.008}"
CLIP_RANGE="${CLIP_RANGE:-0.12}"
TARGET_KL="${TARGET_KL:-0.03}"
AUX_COEF="${AUX_COEF:-0.10}"
EVAL_GAMES="${EVAL_GAMES:-100}"
EVALUATE=1
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: scripts/run_validation_finetune.sh [--dry-run] [--skip-evaluation]

Runs three independent arms from BASE_MODEL:
  epochs2  - n_epochs 2 instead of the control's 1
  aux0     - auxiliary belief loss disabled
  sparse   - terminal-only rewards

Set BASE_MODEL to a checkpoint compatible with the current Observation/Policy
architecture. A validation manifest is required for automatic evaluation.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --skip-evaluation) EVALUATE=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 2 ;;
  esac
  shift
done

for required in "$BASE_MODEL" "$DECK" "$LEAGUE_POOL"; do
  [ -f "$required" ] || { echo "Missing required file: $required" >&2; exit 1; }
done
if [ "$EVALUATE" = 1 ] && [ ! -f "$VALIDATION_FILE" ]; then
  echo "Missing validation manifest: $VALIDATION_FILE" >&2
  echo "Create it first with scripts/build_validation_manifest.py or use --skip-evaluation." >&2
  exit 1
fi

BASE_STEM="$(basename "${BASE_MODEL%.zip}")"
TARGET_PREFIX="models/${BASE_STEM}_ft"
declare -a TARGETS=()

run_arm() {
  local name="$1" epochs="$2" aux="$3" sparse="$4"
  local target="${TARGET_PREFIX}_${name}.zip"
  TARGETS+=("$target")
  if [ -f "$target" ]; then
    echo "Refusing to overwrite existing arm: $target" >&2
    exit 1
  fi
  local cmd=(
    python src/train.py --deck "$DECK" --model-name "$target" --continue-existing
    --opp-deck "$DECK" --opp-pool "$LEAGUE_POOL" --timesteps "$STEPS"
    --num-envs "$NUM_ENVS" --n-steps "$N_STEPS" --batch-size "$BATCH_SIZE"
    --n-epochs "$epochs" --lr "$LR" --ent-coef "$ENT_COEF"
    --clip-range "$CLIP_RANGE" --target-kl "$TARGET_KL" --aux-coef "$aux"
    --rotate-perspective
  )
  [ "$sparse" = 0 ] || cmd+=(--sparse-rewards)
  echo "Fine-tune arm $name: epochs=$epochs aux=$aux sparse=$sparse"
  printf ' %q' "${cmd[@]}"; echo
  [ "$DRY_RUN" = 0 ] || return 0
  cp "$BASE_MODEL" "$target"
  "${cmd[@]}"
}

# The frozen parent is the control. Each arm isolates one change from it.
run_arm epochs2 2 "$AUX_COEF" 0
run_arm aux0 1 0 0
run_arm sparse 1 "$AUX_COEF" 1

if [ "$EVALUATE" = 1 ]; then
  echo "Running automatic validation for all fine-tune arms."
  eval_cmd=(python -m src.evaluation_worker --bot-id "${BASE_STEM}_fine_tune" --games "$EVAL_GAMES" --holdout-file "$VALIDATION_FILE")
  for target in "${TARGETS[@]}"; do eval_cmd+=(--model "$target"); done
  printf ' %q' "${eval_cmd[@]}"; echo
  [ "$DRY_RUN" = 0 ] || exit 0
  "${eval_cmd[@]}"
fi
