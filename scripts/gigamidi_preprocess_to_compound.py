#!/usr/bin/env python3
"""
GigaMIDI Preprocess to Compound Tokens

Runs midi-preprocess.py on GigaMIDI data to convert MIDI to compound tokens.

Supports TWO folder structures:
1. Flat hex folders (NEW - from gigamidi_create_subset_unified.py):
    input/
        0/
        1/
        ...
        d/
        e/

2. Nested train/valid/test structure (OLD - from restructured downloads):
    input/
        train/
            0/
        valid/
            e/
        test/
            f/

Usage:
    python scripts/gigamidi_preprocess_to_compound.py \
        --input data/gigamidi_s1_10pct_random_from_all/ \
        --workers 16
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

    # Detect folder structure
    # Check if flat hex structure (has 0-f folders directly)
    # or nested (has train/valid/test subdirs)
    has_flat_structure = (input_dir / "0").exists()
    has_nested_structure = (input_dir / "train").exists()

    if has_flat_structure:
        print("Detected: FLAT hex folder structure")
        # Process all hex folders directly
        hex_folders = []
        for hex_char in "0123456789abcdef":
            hex_path = input_dir / hex_char
            if hex_path.exists():
                hex_folders.append(hex_char)
        print(f"Hex folders found: {hex_folders}")
        print("-" * 60)
        sys.stdout.flush()

        # Process each hex folder
        for hex_char in hex_folders:
            hex_dir = input_dir / hex_char
            print(f"Processing {hex_char}/...")

            cmd = [
                sys.executable,
                str(TRAIN_DIR / "midi-preprocess.py"),
                str(hex_dir),
                "--workers",
                str(args.workers),
            ]

            print(f"  Running: {' '.join(cmd)}")

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

            print(f"  {hex_char}/ complete")
            sys.stdout.flush()

    elif has_nested_structure:
        print("Detected: NESTED train/valid/test structure")
        # Process train/valid/test split directories
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

            cmd = [
                sys.executable,
                str(TRAIN_DIR / "midi-preprocess.py"),
                str(split_dir),
                "--workers",
                str(args.workers),
            ]

            print(f"  Running: {' '.join(cmd)}")

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
    else:
        print(
            "ERROR: Unknown folder structure. Expected flat hex folders (0-f) or nested train/valid/test"
        )
        sys.exit(1)

    print("-" * 60)
    print("Preprocessing complete!")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
