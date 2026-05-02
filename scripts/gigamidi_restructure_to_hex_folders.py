#!/usr/bin/env python3
"""
GigaMIDI Restructure to Hex Folders

Restructures downloaded MIDI files into hex folders to match Lakh MIDI convention.

Usage:
    python scripts/gigamidi_restructure_to_hex_folders.py \
        --input data/gigamidi_s1_10pct_random_from_all/ \
        --output data/gigamidi_s1_10pct_random_from_all_structured/ \
        --train_split_chars 0123456789abcd \
        --valid_split_chars "" \
        --test_split_chars ""

The script expects flat MIDI files (md5.mid) and reorganizes them into:
    output/
        train/
            0/      # train split (md5 starts with 0-9, a-d)
            1/
            ...
            d/
        valid/   # optional: when valid_split_chars is set
            e/
        test/   # optional: when test_split_chars is set
            f/

Split assignment is based on first char of md5:
    - 0-9, a-d -> train/
    - e -> valid/  
    - f -> test/

You can customize which hex chars map to which split using:
    --train_split_chars 0123456789abcd
    --valid_split_chars e
    --test_split_chars f
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Restructure GigaMIDI to hex folder structure"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with flat MIDI files",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory with hex folder structure",
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
    parser.add_argument(
        "--parallel",
        type=int,
        default=16,
        help="Number of parallel workers (default: 16)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=False)

    print("=" * 60)
    print("GigaMIDI Restructure to Hex Folders")
    print("=" * 60)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Train chars: {args.train_split_chars}")
    print(f"Valid chars: {args.valid_split_chars}")
    print(f"Test chars: {args.test_split_chars}")
    print("-" * 60)

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # Create split directories upfront
    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"
    test_dir = output_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Create hex subdirectories
    for char in args.train_split_chars + args.valid_split_chars + args.test_split_chars:
        (train_dir / char).mkdir(parents=True, exist_ok=True)
        (valid_dir / char).mkdir(parents=True, exist_ok=True)
        (test_dir / char).mkdir(parents=True, exist_ok=True)

    # Find all MIDI files
    mid_files = list(input_dir.glob("*.mid"))
    midi_files = list(input_dir.glob("*.midi"))
    all_files = mid_files + midi_files

    print(f"Found {len(all_files):,} MIDI files")
    print("-" * 60)
    sys.stdout.flush()

    # Track counts
    train_count = 0
    valid_count = 0
    test_count = 0
    skip_count = 0

    for filepath in all_files:
        md5 = filepath.stem  # filename without extension

        if len(md5) < 1:
            skip_count += 1
            continue

        first_char = md5[0].lower()

        # Determine split
        if first_char in args.train_split_chars:
            target_base = train_dir
            train_count += 1
        elif first_char in args.valid_split_chars:
            target_base = valid_dir
            valid_count += 1
        elif first_char in args.test_split_chars:
            target_base = test_dir
            test_count += 1
        else:
            skip_count += 1
            continue

        # Create hex subdirectory and symlink
        hex_dir = target_base / first_char
        target_path = hex_dir / filepath.name

        if not target_path.exists():
            os.symlink(filepath.resolve(), target_path)

        if (train_count + valid_count + test_count) % 5000 == 0:
            print(
                f"  Processed: {train_count + valid_count + test_count:,} / {len(all_files):,}"
            )
            sys.stdout.flush()

    print("-" * 60)
    print("Restructure complete!")
    print(f"  Train:  {train_count:,} files in {args.train_split_chars}")
    print(f"  Valid:  {valid_count:,} files in {args.valid_split_chars}")
    print(f"  Test:   {test_count:,} files in {args.test_split_chars}")
    print(f"  Skipped: {skip_count:,} files")
    print(f"  Output: {output_dir}")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
