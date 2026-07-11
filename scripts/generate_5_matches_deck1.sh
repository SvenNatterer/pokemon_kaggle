#!/bin/bash
cd "$(dirname "$0")/.." || exit 1
source venv/bin/activate
mkdir -p replays/deck1_vs_deck1

for i in {1..5}
do
   echo "Generating Match $i for Deck 1..."
   python tests/test_sub/src/generate_replay.py \
     --deck-a decks/deck_1.csv \
     --model-a backup/top10_manual/ppo_deck_1.zip \
     --deck-b decks/deck_1.csv \
     --model-b backup/top10_manual/ppo_deck_1.zip \
     --out replays/deck1_vs_deck1/match_$i.json
done
echo "All 5 matches generated in replays/deck1_vs_deck1/"
