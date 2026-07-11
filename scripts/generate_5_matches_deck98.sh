#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
source venv/bin/activate
mkdir -p replays/deck98_vs_deck98
MODEL_PATH="models/ppo_v4_deck_98.zip"
if [ ! -f "$MODEL_PATH" ]; then
  MODEL_PATH="models/models/ppo_v4_deck_98.zip"
fi

for i in {1..5}
do
   echo "Generating Match $i with trained model for Deck 98..."
   python src/generate_replay.py \
     --deck-a decks/deck_98.csv \
     --model-a "$MODEL_PATH" \
     --deck-b decks/deck_98.csv \
     --model-b "$MODEL_PATH" \
     --out replays/deck98_vs_deck98/match_$i.json
done
echo "All 5 matches generated in replays/deck98_vs_deck98/"
