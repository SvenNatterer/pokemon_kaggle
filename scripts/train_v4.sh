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
TIMESTEPS=1000000
DECK="decks/deck_98.csv"
MODEL_NAME="models/ppo_v4_deck_98.zip"
NUM_ENVS=8

echo "Training for $TIMESTEPS steps..."
echo "Deck: $DECK"
echo "Model will be saved to: $MODEL_NAME"
echo ""

# Start the training
python src/train.py \
    --deck "$DECK" \
    --model-name "$MODEL_NAME" \
    --timesteps "$TIMESTEPS" \
    --num-envs "$NUM_ENVS"

echo "========================================"
echo "✅ Training Completed!"
echo "Model saved to: $MODEL_NAME"
echo "========================================"
