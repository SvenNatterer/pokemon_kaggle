#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Array of deck names and model names
declare -a decks=("decks/deck_1.csv" "decks/deck_2.csv" "decks/deck_3.csv" "decks/deck_4.csv" "decks/deck_5.csv" "decks/deck_6.csv" "decks/deck_7.csv" "decks/deck_8.csv")
declare -a models=("ppo_deck_1" "ppo_deck_2" "ppo_deck_3" "ppo_deck_4" "ppo_deck_5" "ppo_deck_6" "ppo_deck_7" "ppo_deck_8")

# Number of timesteps per agent
TIMESTEPS=25000

# Number of parallel CPU workers (an deinen Mac anpassen, 8 ist oft optimal)
NUM_ENVS=8

for i in "${!decks[@]}"; do
    DECK="${decks[$i]}"
    MODEL="${models[$i]}"
    
    echo "=========================================================="
    echo "Training agent on $DECK as $MODEL with $NUM_ENVS parallel envs"
    echo "=========================================================="
    
    python src/train.py --deck "$DECK" --model-name "$MODEL" --timesteps $TIMESTEPS --num-envs $NUM_ENVS
done

echo "All training runs completed!"
