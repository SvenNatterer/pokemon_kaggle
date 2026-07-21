#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.." || exit 1

COMPETITION="pokemon-tcg-ai-battle"
ARCHIVE="${1:-}"

if [[ -z "$ARCHIVE" ]]; then
  for candidate in artifacts/submissions/*.tar.gz; do
    [[ -f "$candidate" ]] || continue
    if [[ -z "$ARCHIVE" || "$candidate" -nt "$ARCHIVE" ]]; then
      ARCHIVE="$candidate"
    fi
  done
fi

if [[ -z "$ARCHIVE" || ! -f "$ARCHIVE" ]]; then
  echo "No submission archive found in artifacts/submissions/." >&2
  exit 1
fi

if [[ "$ARCHIVE" != *.tar.gz ]]; then
  echo "Submission archive must end in .tar.gz: $ARCHIVE" >&2
  exit 1
fi

if ! tar -tzf "$ARCHIVE" main.py deck.csv ppo_pokemon_final.zip >/dev/null; then
  echo "Submission archive is invalid or missing main.py, deck.csv, or ppo_pokemon_final.zip: $ARCHIVE" >&2
  exit 1
fi

if [[ ! -x venv/bin/kaggle ]]; then
  echo "Kaggle CLI not found at venv/bin/kaggle." >&2
  exit 1
fi

MESSAGE="${2:-Direct API submission: $(basename "$ARCHIVE")}"

echo "Submitting $ARCHIVE directly to $COMPETITION..."
venv/bin/kaggle competitions submit \
  -c "$COMPETITION" \
  -f "$ARCHIVE" \
  -m "$MESSAGE"
