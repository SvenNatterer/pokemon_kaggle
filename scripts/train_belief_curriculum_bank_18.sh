#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

source venv/bin/activate || true
export PYTHONPATH="src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

DECK="decks/deck_bank/bank_18.csv"
MODEL_NAME="models/ppo_belief_deck_bank_18.zip"
BELIEF_DIM="${BELIEF_DIM:-64}"
TIMESTEPS_STAGE2="${TIMESTEPS_STAGE2:-500000}"
TIMESTEPS_STAGE3="${TIMESTEPS_STAGE3:-250000}"

echo "========================================================"
echo " Stage 2: Frozen Self-Play (Bank 18 against itself)"
echo "========================================================"
# Wir frieren das Modell ein (indem wir es als --opp-model übergeben) und trainieren es gegen sich selbst weiter.
python src/train.py \
  --deck "$DECK" \
  --model-name "$MODEL_NAME" \
  --opp-deck "$DECK" \
  --opp-model "$MODEL_NAME" \
  --timesteps "$TIMESTEPS_STAGE2" \
  --num-envs 8 \
  --checkpoint-interval 100000 \
  --keep-checkpoints 2 \
  --n-epochs 4 \
  --clip-range 0.2 \
  --batch-size 1024 \
  --belief-actor \
  --belief-dim "$BELIEF_DIM" \
  --aux-coef 0.1 \
  --lr 1e-5 \
  --ent-coef 0.005

echo "========================================================"
echo " Stage 3: Kaggle-Anchor Training"
echo "========================================================"
# Wir trainieren kurze Blöcke gegen die stärksten Kaggle-Decks, um nicht zu überfitten.

ANCHORS=(
  "decks/deck_1.csv:models/ppo_deck_1.zip"
  "decks/deck_bank/bank_47.csv:models/ppo_v4_deck_bank_47.zip"
)

for ANCHOR in "${ANCHORS[@]}"; do
  OPP_DECK="${ANCHOR%%:*}"
  OPP_MODEL="${ANCHOR##*:}"

  echo ">>> Training against Kaggle-Anchor: $OPP_MODEL"
  python src/train.py \
    --deck "$DECK" \
    --model-name "$MODEL_NAME" \
    --opp-deck "$OPP_DECK" \
    --opp-model "$OPP_MODEL" \
    --timesteps "$TIMESTEPS_STAGE3" \
    --num-envs 8 \
    --checkpoint-interval 100000 \
    --keep-checkpoints 2 \
    --n-epochs 4 \
    --clip-range 0.2 \
    --batch-size 1024 \
    --belief-actor \
    --belief-dim "$BELIEF_DIM" \
    --aux-coef 0.1 \
    --lr 1e-5 \
    --ent-coef 0.005
done

echo "========================================================"
echo " Stage 4: Evaluation"
echo "========================================================"

echo "Evaluierung gegen die Overnight-Referenzen..."
python scripts/evaluate_overnight_references.py \
  --candidate "$MODEL_NAME" \
  --games 50

echo "Curriculum Complete!"
