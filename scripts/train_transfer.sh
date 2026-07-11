#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

if [ "$#" -lt 2 ]; then
    echo "Usage: ./scripts/train_transfer.sh <source_model_zip> <target_deck_csv_1> [target_deck_csv_2] ..."
    echo "Example: ./scripts/train_transfer.sh models/ppo_v4_deck_4_frozen.zip decks/deck_9.csv decks/deck_12.csv"
    exit 1
fi

SOURCE_MODEL="$1"
shift

if [ ! -f "$SOURCE_MODEL" ]; then
    echo "Error: Source model not found: $SOURCE_MODEL"
    exit 1
fi

# Activate virtual environment once for all runs
source venv/bin/activate || true

for TARGET_DECK in "$@"; do
    if [ ! -f "$TARGET_DECK" ]; then
        echo "Error: Target deck not found: $TARGET_DECK, skipping..."
        continue
    fi

    # Extract deck number from target deck (e.g. decks/deck_9.csv -> 9)
    DECK_NUM=$(basename "$TARGET_DECK" .csv | sed -E 's/^deck_//')
    TARGET_MODEL_NAME="ppo_v4_deck_${DECK_NUM}"
    TARGET_MODEL_FILE="models/${TARGET_MODEL_NAME}.zip"

    echo ""
    echo "================================================="
    echo " TRANSFER LEARNING: DECK $DECK_NUM"
    echo " Source Model: $SOURCE_MODEL"
    echo " Target Deck:  $TARGET_DECK"
    echo " Target Model: $TARGET_MODEL_FILE"
    echo "================================================="

    # Transfer the brain if it doesn't already exist
    if [ ! -f "$TARGET_MODEL_FILE" ]; then
        echo "Copying source weights to create new model..."
        cp "$SOURCE_MODEL" "$TARGET_MODEL_FILE"
    else
        echo "Warning: $TARGET_MODEL_FILE already exists! Resuming training on existing file."
    fi

    STAGE1_OPP_MODEL="models/${TARGET_MODEL_NAME}_opp_stage1.zip"
    echo "Creating frozen snapshot for Stage 1..."
    cp "$TARGET_MODEL_FILE" "$STAGE1_OPP_MODEL"

    echo "=== Stage 1: Initial Adaptation (500k steps) ==="
    python src/train.py \
        --deck "$TARGET_DECK" \
        --model-name "$TARGET_MODEL_NAME" \
        --opp-deck "$TARGET_DECK" \
        --opp-model "$STAGE1_OPP_MODEL" \
        --timesteps 500000 \
        --checkpoint-interval 100000 \
        --num-envs 6 \
        --lr 5e-5 \
        --ent-coef 0.005 \
        --n-epochs 4 \
        --sparse-rewards

    STAGE2_OPP_MODEL="models/${TARGET_MODEL_NAME}_opp_stage2.zip"
    echo "Creating frozen snapshot for Stage 2..."
    cp "$TARGET_MODEL_FILE" "$STAGE2_OPP_MODEL"

    echo "=== Stage 2: Deep Self-Play (1M steps) ==="
    python src/train.py \
        --deck "$TARGET_DECK" \
        --model-name "$TARGET_MODEL_NAME" \
        --opp-deck "$TARGET_DECK" \
        --opp-model "$STAGE2_OPP_MODEL" \
        --timesteps 1000000 \
        --checkpoint-interval 200000 \
        --num-envs 6 \
        --lr 3e-5 \
        --ent-coef 0.003 \
        --n-epochs 4 \
        --sparse-rewards
        
    echo "Successfully completed two-stage training for $TARGET_DECK!"
    echo ""
done
