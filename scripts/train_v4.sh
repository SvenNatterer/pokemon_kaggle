#!/bin/bash
set -e

# ==============================================================================
# v4 "Perfect Representation" Training Script
# ==============================================================================
# This script trains a completely fresh PPO model using the 1500-dimensional 
# observation space.

echo "========================================"
echo "Starting v4 Training (1500 Dimensions)"
echo "========================================"

# Make sure we use the virtual environment
source venv/bin/activate

# Set PYTHONPATH to include src
export PYTHONPATH="src:$PYTHONPATH"

# Configuration
DECK="decks/deck_98.csv"
MODEL_NAME="models/ppo_v4_deck_98.zip"
NUM_ENVS=8
CHECKPOINT_INTERVAL=250000
KEEP_CHECKPOINTS=2

echo "Training endlessly. Press Ctrl+C to stop after the latest checkpoint/save."
echo "Deck: $DECK"
echo "Model will be saved to: $MODEL_NAME"
echo "Rotating checkpoints: ${MODEL_NAME%.zip}_checkpoint_1.zip / ${MODEL_NAME%.zip}_checkpoint_2.zip"
echo ""

# Start the training
python src/train.py \
    --deck "$DECK" \
    --model-name "$MODEL_NAME" \
    --opp-deck "decks/deck_98.csv" \
    --endless \
    --num-envs "$NUM_ENVS" \
    --checkpoint-interval "$CHECKPOINT_INTERVAL" \
    --keep-checkpoints "$KEEP_CHECKPOINTS" \
    --n-epochs 4 \
    --clip-range 0.2 \
    --batch-size 1024

echo "========================================"
echo "Training stopped."
echo "Model saved to: $MODEL_NAME"
echo "========================================"
