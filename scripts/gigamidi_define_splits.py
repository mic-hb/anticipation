#!/usr/bin/env python3
"""
GigaMIDI Define Splits

Defines train/valid/test splits from tokenized event files by moving/renaming them.

This follows the LakhMIDI convention:
- tokenized-events-0 through d -> train
- tokenized-events-e -> valid
- tokenized-events-f -> test

Usage:
    python scripts/gigamidi_define_splits.py \
        --input data/gigamidi_s1_10pct_random_from_all_structured/ \
        --train_output data/gigamidi_s1_train.txt \
        --valid_output data/gigamidi_s1_valid.txt \
        --test_output data/gigamidi_s1_test.txt

Features:
- Progress bar with tqdm
- Time estimates
- Summary statistics
"""

import argparse
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Define train/valid/test splits from tokenized files"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with tokenized-events-*.txt files",
    )
    parser.add_argument(
        "--train_output",
        type=str,
        default=None,
        help="Output file for training data (or - for concatenation mode)",
    )
    parser.add_argument(
        "--valid_output",
        type=str,
        default=None,
        help="Output file for validation data",
    )
    parser.add_argument(
        "--test_output",
        type=str,
        default=None,
        help="Output file for test data",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        default=False,
        help="Concatenate train splits into one file instead of moving",
    )
    parser.add_argument(
        "--train_split_chars",
        type=str,
        default="0123456789abcd",
        help="Hex chars for train split (default: 0123456789abcd)",
    )
    parser.add_argument(
        "--valid_split_chars",
        type=str,
        default="e",
        help="Hex chars for valid split (default: e)",
    )
    parser.add_argument(
        "--test_split_chars",
        type=str,
        default="f",
        help="Hex chars for test split (default: f)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    start_time = time.time()

    print("=" * 70)
    print("GigaMIDI Define Splits")
    print("=" * 70)
    print(f"Input:   {args.input}")
    print(f"Train:   {args.train_split_chars}")
    print(f"Valid:   {args.valid_split_chars}")
    print(f"Test:    {args.test_split_chars}")
    print(f"Mode:    {'concatenate' if args.concat else 'move'}")
    print("-" * 70)
    sys.stdout.flush()

    input_dir = Path(args.input)

    # Stage 1: Find tokenized files
    print("\n[Stage 1/3] Finding tokenized files...")
    stage_start = time.time()

    train_files = []
    valid_file = None
    test_file = None

    for char in args.train_split_chars:
        f = input_dir / f"tokenized-events-{char}.txt"
        if f.exists():
            train_files.append(f)

    valid_file = input_dir / f"tokenized-events-{args.valid_split_chars}.txt"
    if not valid_file.exists():
        valid_file = None

    test_file = input_dir / f"tokenized-events-{args.test_split_chars}.txt"
    if not test_file.exists():
        test_file = None

    print(f"  Train files: {len(train_files)}")
    print(f"  Valid file exists: {valid_file is not None and valid_file.exists()}")
    print(f"  Test file exists: {test_file is not None and test_file.exists()}")
    sys.stdout.flush()

    # Stage 2: Process files
    print("\n[Stage 2/3] Defining splits...")
    stage_start = time.time()
    print("-" * 70)

    train_lines = 0
    valid_lines = 0
    test_lines = 0

    if args.concat:
        # Concatenate train files
        print(f"  Concatenating {len(train_files)} train files...")
        train_output_path = (
            Path(args.train_output) if args.train_output else input_dir / "train.txt"
        )

        with tqdm(total=len(train_files), desc="Train concat", unit="files") as pbar:
            with open(train_output_path, "w") as out_f:
                for f in train_files:
                    with open(f, "r") as in_f:
                        for line in in_f:
                            out_f.write(line)
                            train_lines += 1
                    pbar.update(1)

        print(f"  Train lines: {train_lines:,}")
    else:
        # Move each file individually
        print(f"  Renaming {len(train_files)} train files...")
        with tqdm(total=len(train_files), desc="Train rename", unit="files") as pbar:
            for f in train_files:
                # Move to train-ordered.txt
                new_name = f.with_name("train-ordered.txt")
                if f != new_name:
                    f.rename(new_name)
                train_lines += 1
                pbar.update(1)

    # Handle validation
    if valid_file and valid_file.exists():
        if args.valid_output:
            print(f"  Saving validation to {args.valid_output}...")
            dest = Path(args.valid_output)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(valid_file, "r") as src:
                with open(dest, "w") as dst:
                    for line in src:
                        dst.write(line)
                        valid_lines += 1
        else:
            new_name = valid_file.with_name("valid.txt")
            valid_file.rename(new_name)
            valid_lines = sum(1 for _ in open(new_name))

    # Handle test
    if test_file and test_file.exists():
        if args.test_output:
            print(f"  Saving test to {args.test_output}...")
            dest = Path(args.test_output)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(test_file, "r") as src:
                with open(dest, "w") as dst:
                    for line in src:
                        dst.write(line)
                        test_lines += 1
        else:
            new_name = test_file.with_name("test.txt")
            test_file.rename(new_name)
            test_lines = sum(1 for _ in open(new_name))

    print(f"\n  Train lines:  {train_lines:,}")
    print(f"  Valid lines: {valid_lines:,}")
    print(f"  Test lines:  {test_lines:,}")
    sys.stdout.flush()

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Train:  {train_lines:,}")
    print(f"  Valid:  {valid_lines:,}")
    print(f"  Test:   {test_lines:,}")
    print(f"  Total:  {train_lines + valid_lines + test_lines:,}")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
