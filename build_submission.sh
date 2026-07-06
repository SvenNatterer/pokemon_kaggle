#!/usr/bin/env bash
set -e

echo "Creating submission archive..."

# Create a temporary directory for building the submission
mkdir -p submission_build
cp submission/main.py submission_build/
cp models/ppo_pokemon_final.zip submission_build/
cp -r src submission_build/
cp -r pokemon-tcg-ai-battle/sample_submission/sample_submission/cg submission_build/
cp pokemon-tcg-ai-battle/sample_submission/sample_submission/deck.csv submission_build/

# Create the tar.gz archive
cd submission_build
tar -czvf ../submission.tar.gz *
cd ..

# Clean up
rm -rf submission_build

echo "Submission archive created: submission.tar.gz"
