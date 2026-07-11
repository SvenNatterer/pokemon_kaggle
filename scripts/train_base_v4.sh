#!/bin/bash
set -e

# Activate virtual environment automatically
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo "Virtual environment activated."
fi

echo "================================================================"
echo " Starting v4 Base Brain Training Curriculum"
echo "================================================================"

# Stage 1: Fahrschule (Mirror without opp-model)
# echo ""
# echo "=== Stage 1: Fahrschule (100k - 250k steps against dumb deterministic option 0) ==="
# python src/train.py --deck decks/deck_base_v4.csv \
#     --model-name ppo_v4_base_brain \
#     --timesteps 250000 \
#     --checkpoint-interval 125000 \
#     --lr 3e-4 --ent-coef 0.03 --n-epochs 4

# Stage 2: Frozen Self (Base vs Base)
# echo ""
# echo "=== Stage 2: Frozen Self (500k - 1M steps against own snapshot) ==="
# # Freeze current checkpoint as opponent
# cp models/ppo_v4_base_brain.zip models/ppo_v4_base_brain_frozen.zip
# python src/train.py --deck decks/deck_base_v4.csv \
#     --model-name ppo_v4_base_brain \
#     --opp-deck decks/deck_base_v4.csv \
#     --opp-model models/ppo_v4_base_brain_frozen.zip \
#     --timesteps 1000000 \
#     --checkpoint-interval 250000 \
#     --lr 3e-4 --ent-coef 0.03 --n-epochs 4

# Stage 3: Training against weak v4 checkpoints
# echo ""
# echo "=== Stage 3: Weak Opponents (deck_98) ==="
# python src/train.py --deck decks/deck_base_v4.csv \
#     --model-name ppo_v4_base_brain \
#     --opp-deck decks/deck_98.csv \
#     --opp-model models/ppo_v4_deck_98.zip \
#     --timesteps 500000 \
#     --checkpoint-interval 250000 \
#     --lr 1e-4 --ent-coef 0.015 --n-epochs 4

# Stage 4: Training against deck_16 and deck_1
# echo ""
# echo "=== Stage 4: Scaling up Difficulty (deck_16 then deck_1) ==="
# python src/train.py --deck decks/deck_base_v4.csv \
#     --model-name ppo_v4_base_brain \
#     --opp-deck decks/deck_16.csv \
#     --opp-model models/ppo_deck_16.zip \
#     --timesteps 500000 \
#     --checkpoint-interval 250000 \
#     --lr 1e-4 --ent-coef 0.015 --n-epochs 4

# python src/train.py --deck decks/deck_base_v4.csv \
#     --model-name ppo_v4_base_brain \
#     --opp-deck decks/deck_1.csv \
#     --opp-model models/ppo_deck_1.zip \
#     --timesteps 500000 \
#     --checkpoint-interval 250000 \
#     --lr 1e-4 --ent-coef 0.015 --n-epochs 4

# Stage 5: Advanced Mixed Training (Endless)
echo ""
echo "=== Stage 5: Endless Mixed Training (50% mirror, 30% similar, 20% strong) ==="
echo "Starting continuous loop of mixed training..."

# Protect the model from Stage 4 by creating a separate model for endless training
if [ ! -f "models/ppo_v4_base_brain_endless.zip" ]; then
    echo "Creating ppo_v4_base_brain_endless.zip from ppo_v4_base_brain.zip"
    cp models/ppo_v4_base_brain.zip models/ppo_v4_base_brain_endless.zip
    ln -sf ppo_v4_base_brain_endless.zip models/ppo_v4_deck_4_endless.zip
fi

ITER=1
while true; do
    echo "================================================="
    echo " ENDLESS TRAINING ITERATION $ITER"
    echo "================================================="
    
    echo "--- [50%] Frozen Mirror Training ---"
    # Update frozen snapshot before mirror
    cp models/ppo_v4_base_brain_endless.zip models/ppo_v4_base_brain_frozen.zip
    python src/train.py --deck decks/deck_base_v4.csv \
        --model-name ppo_v4_base_brain_endless \
        --opp-deck decks/deck_base_v4.csv \
        --opp-model models/ppo_v4_base_brain_frozen.zip \
        --timesteps 500000 \
        --checkpoint-interval 250000 \
        --num-envs 6 \
        --lr 5e-5 --ent-coef 0.005 --n-epochs 4
    
    echo "--- [30%] Similar Elo Training (deck_1) ---"
    python src/train.py --deck decks/deck_base_v4.csv \
        --model-name ppo_v4_base_brain_endless \
        --opp-deck decks/deck_1.csv \
        --opp-model models/ppo_deck_1.zip \
        --timesteps 300000 \
        --checkpoint-interval 150000 \
        --num-envs 6 \
        --lr 5e-5 --ent-coef 0.005 --n-epochs 4
        
    echo "--- [20%] Strong Opponent Training (deck_9) ---"
    python src/train.py --deck decks/deck_base_v4.csv \
        --model-name ppo_v4_base_brain_endless \
        --opp-deck decks/deck_9.csv \
        --opp-model models/ppo_deck_9.zip \
        --timesteps 200000 \
        --checkpoint-interval 100000 \
        --num-envs 6 \
        --lr 5e-5 --ent-coef 0.005 --n-epochs 4
        
    # Save a checkpoint of the completed iteration
    cp models/ppo_v4_base_brain_endless.zip models/ppo_v4_base_brain_endless_iter_${ITER}.zip
    ln -sf ppo_v4_base_brain_endless_iter_${ITER}.zip models/ppo_v4_deck_4_endless_iter_${ITER}.zip
    
    ITER=$((ITER+1))
done
