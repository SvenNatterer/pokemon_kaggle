#!/bin/bash
cd "$(dirname \"$0\")/.." || exit 1
source venv/bin/activate
mkdir -p replays/deck100_vs_deck100

for i in {1..5}
do
   echo "Generating Match $i..."
   python src/generate_replay.py \
     --deck-a decks/deck_100.csv \
     --deck-b decks/deck_100.csv \
     --out replays/deck100_vs_deck100/match_$i.json
done
echo "All 5 matches generated in replays/deck100_vs_deck100/"
