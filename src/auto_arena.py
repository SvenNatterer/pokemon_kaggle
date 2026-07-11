"""Compatibility entry point for the queue-free arena worker."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.arena_worker import main


if __name__ == "__main__":
    raise SystemExit(main())
