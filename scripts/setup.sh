#!/usr/bin/env bash
set -e

cd "$(dirname \"$0\")/.." || exit 1
echo "Creating python virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo "Setup complete! Run 'source venv/bin/activate' to activate the environment."
