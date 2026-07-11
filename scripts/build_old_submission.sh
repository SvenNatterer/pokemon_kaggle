#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.." || exit 1
DECK_NUM="9"
MODEL_FILE="models/ppo_deck_9.zip"
DECK_FILE="decks/deck_9.csv"

echo "Creating OLD submission archive for Deck $DECK_NUM using $MODEL_FILE..."

mkdir -p old_src
git archive 5dce404 src/ | tar -x -C old_src

mkdir -p submission_build_v3
cp submission/main.py submission_build_v3/
cp "$MODEL_FILE" submission_build_v3/ppo_pokemon_final.zip
cp -r old_src/src submission_build_v3/
cp -r venv/lib/python3.12/site-packages/sb3_contrib submission_build_v3/
cp -r venv/lib/python3.12/site-packages/stable_baselines3 submission_build_v3/
cp -r venv/lib/python3.12/site-packages/gymnasium submission_build_v3/
cp -r venv/lib/python3.12/site-packages/farama_notifications submission_build_v3/
cp -r pokemon-tcg-ai-battle/sample_submission/sample_submission/cg submission_build_v3/
cp "$DECK_FILE" submission_build_v3/deck.csv

find submission_build_v3 -name "__pycache__" -type d -exec rm -rf {} +

cd submission_build_v3
tar -czvf ../submission_v3_deck9.tar.gz *
cd ..

rm -rf submission_build_v3
rm -rf old_src

echo "Old submission archive created: submission_v3_deck9.tar.gz"
