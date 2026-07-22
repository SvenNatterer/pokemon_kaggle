#!/usr/bin/env python3
"""Build the minimal source/model bundle consumed by the Kaggle V6 notebook."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "kaggle_bundle" / "pokemon_kaggle_v6_bundle.zip"

MODEL_FILES = (
    "models/ppo_v6_mewtwo_quickwins_devpool.zip",
    "models/foundation/compact_potential/ppo_v6_deck_bank_54_compact_a.zip",
    "models/foundation/compact_potential/ppo_v6_deck_bank_55_compact_b.zip",
    "models/foundation/compact_potential/ppo_v6_deck_bank_56_compact_c.zip",
)

SOURCE_ROOTS = (
    "decks",
    "scripts",
    "src",
)


def included_files() -> list[Path]:
    files = [ROOT / "requirements.txt"]
    for relative_root in SOURCE_ROOTS:
        files.extend(path for path in (ROOT / relative_root).rglob("*") if path.is_file())
    files.extend(ROOT / relative for relative in MODEL_FILES)
    return sorted(
        {
            path
            for path in files
            if "__pycache__" not in path.parts
            and path.suffix != ".pyc"
            and path.name != ".DS_Store"
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--owner-slug",
        help="Optional Kaggle username; writes dataset-metadata.json beside the archive.",
    )
    args = parser.parse_args()

    files = included_files()
    missing = [path.relative_to(ROOT).as_posix() for path in files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing bundle inputs:\n  - " + "\n  - ".join(missing))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in files:
            archive.write(path, Path("pokemon_kaggle") / path.relative_to(ROOT))

    if args.owner_slug:
        metadata = {
            "title": "Pokemon Kaggle V6 Training Bundle",
            "id": f"{args.owner_slug}/pokemon-kaggle-v6-training-bundle",
            "licenses": [{"name": "other"}],
        }
        metadata_path = args.output.parent / "dataset-metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    size_mib = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {args.output} ({size_mib:.1f} MiB, {len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
