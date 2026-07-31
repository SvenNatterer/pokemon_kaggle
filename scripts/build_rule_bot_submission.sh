#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
OUTPUT_FILE="${1:-artifacts/submissions/submission_rule_based_lucario.tar.gz}"
AGENT_FILE="${2:-src/agents/kaggle_bots/mega_lucario_agent.py}"
DECK_FILE="${3:-decks/kaggle_bots/deck_kaggle_mega_lucario.csv}"

if [[ ! -f "$DECK_FILE" ]]; then
  echo "Deck not found: $DECK_FILE" >&2
  exit 1
fi

if [[ ! -f "$AGENT_FILE" ]]; then
  echo "Agent script not found: $AGENT_FILE" >&2
  exit 1
fi

if ! file src/cg/libcg.so | grep -q "ELF 64-bit.*x86-64"; then
  echo "src/cg/libcg.so is not an x86-64 Linux library." >&2
  exit 1
fi

echo "Creating Rule-Based submission archive ($OUTPUT_FILE) for $AGENT_FILE..."
mkdir -p "$(dirname "$OUTPUT_FILE")"

rm -rf submission_build
mkdir -p submission_build
# The official rule bot only needs the game API.  Rewriting this import in the
# staged copy avoids executing src/__init__.py, which imports PPO/TensorBoard.
sed 's/^from src\.cg\.api import /from cg.api import /' \
  "$AGENT_FILE" > submission_build/main.py
cp -r src/cg submission_build/cg
cp "$DECK_FILE" submission_build/deck.csv

find submission_build -name "__pycache__" -type d -exec rm -rf {} +
find submission_build -name "*.pyc" -type f -delete
find submission_build -name ".DS_Store" -type f -delete

cd submission_build
export COPYFILE_DISABLE=1
tar --exclude="._*" -czf "../$OUTPUT_FILE" *
cd ..

rm -rf submission_build

echo "Submission archive created successfully: $OUTPUT_FILE"
