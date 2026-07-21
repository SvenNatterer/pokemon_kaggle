#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1
OUTPUT_FILE="${3:-artifacts/submissions/submission.tar.gz}"
if [[ "$OUTPUT_FILE" != *.tar.gz ]]; then
  echo "Submission archive must end in .tar.gz: $OUTPUT_FILE" >&2
  exit 1
fi

if [[ $# -gt 0 ]]; then
  DECK_NUM="$1"
  MODEL_FILE="${2:-}"
  if [[ -z "$MODEL_FILE" ]]; then
    for candidate in "models/ppo_v6_deck_${DECK_NUM}.zip" "models/ppo_v5b_deck_${DECK_NUM}.zip" "models/ppo_v5_deck_${DECK_NUM}.zip" "models/ppo_belief_deck_${DECK_NUM}.zip" "models/ppo_v4_deck_${DECK_NUM}.zip" "models/ppo_deck_${DECK_NUM}.zip"; do
      if [[ -f "$candidate" ]]; then
        MODEL_FILE="$candidate"
        break
      fi
    done
  fi
else
  MODEL_FILE=$({ ls -t models/ppo_v6_deck_*.zip models/ppo_v5b_deck_*.zip models/ppo_v5_deck_*.zip models/ppo_belief_deck_*.zip models/ppo_v4_deck_*.zip models/ppo_deck_*.zip 2>/dev/null || true; } | head -n 1)
  if [[ -z "$MODEL_FILE" ]]; then
    echo "No supported ppo_v6/ppo_v5b/ppo_v5/ppo_belief/ppo_v4/ppo model found." >&2
    exit 1
  fi
  DECK_NUM=$(basename "$MODEL_FILE" .zip | sed -E 's/^ppo(_v4|_v5|_v5b|_v6|_belief)?_deck_//')
fi

if [[ "$DECK_NUM" == bank_* ]]; then
  DECK_FILE="decks/deck_bank/${DECK_NUM}.csv"
else
  DECK_BASE_NUM=$(echo "$DECK_NUM" | grep -o "^[0-9]*")
  DECK_FILE="decks/deck_${DECK_BASE_NUM}.csv"
fi

if [[ ! -f "$MODEL_FILE" ]]; then
  echo "Model not found: $MODEL_FILE" >&2
  exit 1
fi

if [[ ! -f "$DECK_FILE" ]]; then
  echo "Deck not found: $DECK_FILE" >&2
  exit 1
fi

if ! file src/cg/libcg.so | grep -q "ELF 64-bit.*x86-64"; then
  echo "src/cg/libcg.so is not an x86-64 Linux library." >&2
  exit 1
fi
if ! nm -D src/cg/libcg.so 2>/dev/null | grep " GetV6Observation$" >/dev/null; then
  echo "src/cg/libcg.so does not export GetV6Observation." >&2
  echo "Run scripts/build_linux_v6_engine.sh first." >&2
  exit 1
fi

echo "Creating submission archive for Deck $DECK_NUM using $MODEL_FILE..."
mkdir -p "$(dirname "$OUTPUT_FILE")"

# Create a temporary directory for building the submission
rm -rf submission_build
mkdir -p submission_build
cp submission/main.py submission_build/
cp "$MODEL_FILE" submission_build/ppo_pokemon_final.zip
cp -r src submission_build/
cp -r venv/lib/python3.12/site-packages/sb3_contrib submission_build/
cp -r venv/lib/python3.12/site-packages/stable_baselines3 submission_build/
cp -r venv/lib/python3.12/site-packages/gymnasium submission_build/
cp -r venv/lib/python3.12/site-packages/farama_notifications submission_build/
cp -r src/cg submission_build/cg

cp "$DECK_FILE" submission_build/deck.csv

# Remove local metadata and caches from the upload.
find submission_build -name "__pycache__" -type d -exec rm -rf {} +
find submission_build -name "*.pyc" -type f -delete
find submission_build -name ".DS_Store" -type f -delete

# Create the tar.gz archive
cd submission_build
tar -czf "../$OUTPUT_FILE" *
cd ..

# Clean up
rm -rf submission_build

echo "Submission archive created: $OUTPUT_FILE"
echo "Upload directly with: scripts/submit_latest_submission.sh \"$OUTPUT_FILE\""
