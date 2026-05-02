#!/usr/bin/env python3
"""
GigaMIDI Preprocess to Compound Tokens

Runs midi-preprocess.py on restructured GigaMIDI data to convert MIDI to compound tokens.

Usage:
    python scripts/gigamidi_preprocess_to_compound.py \
        --input data/gigamidi_s1_10pct_random_from_all_structured/

This runs the standard anticipation preprocessing on the hex folder structure.
It expects:
    input/
        train/
            0/
            1/
            ...
        valid/
            e/
        test/
            f/

And produces:
    input/
        train/
            0/
                file.mid
                file.mid.compound.txt
        ...
"""

import argparse
import os
import sys
from pathlib import Path

# Run from anticipation package directory
SCRIPT_DIR = Path(__file__).parent
ANTICIPATION_DIR = SCRIPT_DIR.parent
TRAIN_DIR = ANTICIPATION_DIR / "train"


def main():
    parser = argparse.ArgumentParser(description="Preprocess GigaMIDI MIDI files")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with hex folder structure",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=False)

    print("=" * 60)
    print("GigaMIDI Preprocess Wrapper")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Workers: {args.workers}")
    print("-" * 60)

    input_dir = Path(args.input)

    # Find all splits
    splits = []
    for split in ["train", "valid", "test"]:
        split_path = input_dir / split
        if split_path.exists():
            splits.append(split)

    print(f"Found splits: {splits}")
    print("-" * 60)
    sys.stdout.flush()

    # Run preprocessing on each split
    for split in splits:
        split_dir = input_dir / split
        print(f"Processing {split}/...")

        # Build find command for midi files
        cmd = [
            sys.executable,
            str(TRAIN_DIR / "midi-preprocess.py"),
            str(split_dir),
        ]

        print(f"  Running: {' '.join(cmd)}")

        # Run the preprocessing script
        os.chdir(ANTICIPATION_DIR)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ANTICIPATION_DIR)

        import subprocess

        result = subprocess.run(
            cmd,
            env=env,
            cwd=str(ANTICIPATION_DIR),
        )

        if result.returncode != 0:
            print(f"  Error: preprocessing failed with code {result.returncode}")
            sys.exit(result.returncode)

        print(f"  {split}/ complete")
        sys.stdout.flush()

    print("-" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
