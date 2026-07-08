import sys
import os

# Ensure the root directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.arena_utils import run_tournament

if __name__ == "__main__":
    run_tournament()
