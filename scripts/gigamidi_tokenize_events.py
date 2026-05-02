#!/usr/bin/env python3
"""
GigaMIDI Tokenize to Event Tokens

Runs tokenize-lakh.py on preprocessed GigaMIDI data to convert compound tokens to event tokens.

Usage:
    python scripts/gigamidi_tokenize_events.py \
        --input data/gigamidi_s1_10pct_random_from_all_structured/

This runs the standard tokenization on hex folder structure.
It expects:
    input/
        train/
            0/
                file.mid
                file.mid.compound.txt
            ...
        valid/
            e/
        test/
            f/

And produces:
    input/
        tokenized-events-train.txt
        tokenized-events-valid.txt
        tokenized-events-test.txt
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
    parser = argparse.ArgumentParser(description="Tokenize GigaMIDI compound files")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with preprocessed hex folder structure",
    )
    parser.add_argument(
        "--augment",
        type=int,
        default=1,
        help="Augmentation factor (default: 1)",
    )
    parser.add_argument(
        "--interarrival",
        action="store_true",
        help="Use interarrival-time encoding (default: arrival-time)",
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
    print("GigaMIDI Tokenize Wrapper")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Augment: {args.augment}x")
    print(f"Encoding: {'interarrival' if args.interarrival else 'arrival'}")
    print(f"Workers: {args.workers}")
    print("-" * 60)

    input_dir = Path(args.input)

    # Build tokenize-lakh.py command
    cmd = [
        sys.executable,
        str(TRAIN_DIR / "tokenize-lakh.py"),
        str(input_dir),
        "--augment",
        str(args.augment),
    ]

    if args.interarrival:
        cmd.append("--interarrival")

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)
    sys.stdout.flush()

    # Run the tokenization script
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
        print(f"Error: tokenization failed with code {result.returncode}")
        sys.exit(result.returncode)

    print("-" * 60)
    print("Tokenization complete!")
    print(f"  Output files in: {input_dir}")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
