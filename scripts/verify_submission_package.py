#!/usr/bin/env python3
"""Verify built Kaggle submission package archive for P0 Submission Security."""

import argparse
import os
import sys
import subprocess
import tarfile
import zipfile
import tempfile
import importlib.util
from pathlib import Path


def verify_package(archive_path: str) -> bool:
    print(f"=== Verifying Submission Archive: {archive_path} ===")
    
    if not os.path.exists(archive_path):
        print(f"[FAIL] Archive file does not exist: {archive_path}")
        return False
        
    archive_size_mb = os.path.getsize(archive_path) / (1024 * 1024)
    print(f"Archive size: {archive_size_mb:.2f} MB")
    if archive_size_mb > 250.0:
        print(f"[WARNING] Archive size exceeds 250MB ({archive_size_mb:.2f} MB)")
    else:
        print(f"[PASS] Archive size is within limits (< 250MB)")
        
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Extracting to temporary directory: {temp_dir}")
        if archive_path.endswith(".tar.gz") or archive_path.endswith(".tgz"):
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=temp_dir)
        elif archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, "r") as zip_ref:
                zip_ref.extractall(path=temp_dir)
        else:
            print(f"[FAIL] Unsupported archive format: {archive_path}")
            return False

        # Check required files
        required_files = ["main.py", "deck.csv", "ppo_pokemon_final.zip"]
        for rf in required_files:
            fp = os.path.join(temp_dir, rf)
            if not os.path.exists(fp):
                print(f"[FAIL] Missing required file: {rf}")
                return False
            print(f"[PASS] File present: {rf}")

        # Test standalone Python execution of main.py in an isolated subprocess
        print("\nTesting standalone importability of main.py in isolated process...")
        code = (
            "import sys, main; "
            "deck = main.agent({'select': None}); "
            "assert isinstance(deck, list) and len(deck) == 60, f'Invalid deck: {len(deck)} cards'; "
            "print('[PASS] Standalone agent deck selection returned valid 60-card list')"
        )
        sub_env = os.environ.copy()
        sub_env["PYTHONPATH"] = temp_dir
        
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=temp_dir,
            env=sub_env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            print(f"[FAIL] Subprocess main.py execution failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr}")
            return False
        print(proc.stdout.strip())

            
    print("\n[SUCCESS] Submission package verification PASSED completely!")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify Kaggle submission package.")
    parser.add_argument("archive_path", help="Path to submission archive (.tar.gz or .zip)")
    args = parser.parse_args()
    
    success = verify_package(args.archive_path)
    sys.exit(0 if success else 1)
