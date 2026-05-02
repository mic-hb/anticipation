#!/usr/bin/env python3
"""
GigaMIDI Restructure to Hex Folders

Restructures downloaded MIDI files into hex folders to match Lakh MIDI convention.

Usage:
    python scripts/gigamidi_restructure_to_hex_folders.py \
        --input data/gigamidi_s1_10pct_random_from_all/ \
        --output data/gigamidi_s1_10pct_random_from_all_structured/ \
        --train_split_chars 0123456789abcd \
        --valid_split_chars e \
        --test_split_chars f

Features:
- Progress bar with tqdm
- Time estimates
- Symlink-based (no file copy)
- Skips existing symlinks
- Split by first hex char of MD5

Directory structure:
    output/
        train/
            0/      # train split
            1/
            ...
            d/
        valid/   # optional
            e/
        test/   # optional
            f/
"""

import argparse
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Restructure GigaMIDI flat files to hex folder structure"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory with flat MIDI files (md5.mid)",
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
        "--use_symlinks",
        action="store_true",
        default=True,
        help="Use symlinks instead of copying files (default: True)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    start_time = time.time()

    print("=" * 70)
    print("GigaMIDI Restructure to Hex Folders")
    print("=" * 70)
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Train:   {args.train_split_chars or '(disabled)'}")
    print(f"Valid:   {args.valid_split_chars or '(disabled)'}")
    print(f"Test:    {args.test_split_chars or '(disabled)'}")
    print(f"Symlinks: {args.use_symlinks}")
    print("-" * 70)
    sys.stdout.flush()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Stage 1: Find all MIDI files
    print("\n[Stage 1/3] Scanning for MIDI files...")
    stage_start = time.time()

    mid_files = list(input_dir.glob("*.mid"))
    midi_files = list(input_dir.glob("*.midi"))
    all_files = mid_files + midi_files
    total_files = len(all_files)

    scan_time = time.time() - stage_start
    print(f"  Found {total_files:,} MIDI files in {scan_time:.1f}s")
    sys.stdout.flush()

    # Stage 2: Create directory structure
    print("\n[Stage 2/3] Creating directory structure...")
    stage_start = time.time()

    train_dir = output_dir / "train"
    valid_dir = output_dir / "valid"
    test_dir = output_dir / "test"

    if args.train_split_chars:
        train_dir.mkdir(parents=True, exist_ok=True)
        for char in args.train_split_chars:
            (train_dir / char).mkdir(parents=True, exist_ok=True)

    if args.valid_split_chars:
        valid_dir.mkdir(parents=True, exist_ok=True)
        for char in args.valid_split_chars:
            (valid_dir / char).mkdir(parents=True, exist_ok=True)

    if args.test_split_chars:
        test_dir.mkdir(parents=True, exist_ok=True)
        for char in args.test_split_chars:
            (test_dir / char).mkdir(parents=True, exist_ok=True)

    dir_time = time.time() - stage_start
    print(f"  Created directories in {dir_time:.1f}s")
    sys.stdout.flush()

    # Stage 3: Create symlinks
    print("\n[Stage 3/3] Creating symlinks...")
    stage_start = time.time()
    print("-" * 70)

    train_count = 0
    valid_count = 0
    test_count = 0
    skip_count = 0

    with tqdm(
        total=total_files,
        desc="Processing",
        unit="files",
        unit_scale=True,
        unit_divisor=1000,
    ) as pbar:
        for filepath in all_files:
            md5 = filepath.stem

            if len(md5) < 1:
                skip_count += 1
                pbar.update(1)
                continue

            first_char = md5[0].lower()

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
                pbar.update(1)
                continue

            hex_dir = target_base / first_char
            target_path = hex_dir / filepath.name

            if target_path.exists():
                skip_count += 1
                pbar.update(1)
                continue

            if args.use_symlinks:
                try:
                    os.symlink(filepath.resolve(), target_path)
                except OSError:
                    skip_count += 1
            else:
                import shutil

                try:
                    shutil.copy2(filepath, target_path)
                except Exception:
                    skip_count += 1

            pbar.update(1)

            if (train_count + valid_count + test_count) % 5000 == 0:
                elapsed = time.time() - stage_start
                rate = (
                    (train_count + valid_count + test_count) / elapsed
                    if elapsed > 0
                    else 0
                )
                remaining = total_files - (
                    train_count + valid_count + test_count + skip_count
                )
                eta = remaining / rate if rate > 0 else 0
                pbar.set_postfix(
                    {
                        "train": f"{train_count:,}",
                        "valid": f"{valid_count:,}",
                        "test": f"{test_count:,}",
                        "ETA": f"{eta / 60:.1f}m" if eta > 0 else "done",
                    }
                )

    link_time = time.time() - stage_start
    total_linked = train_count + valid_count + test_count
    rate = total_linked / link_time if link_time > 0 else 0
    print(f"\n  Linked in {link_time:.1f}s ({rate:.0f} files/s)")
    sys.stdout.flush()

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Total files scanned:   {total_files:,}")
    print(f"  Train links:      {train_count:,}")
    print(f"  Valid links:     {valid_count:,}")
    print(f"  Test links:      {test_count:,}")
    print(f"  Skipped:        {skip_count:,}")
    print(f"  Total linked:    {total_linked:,}")
    print(f"  Output:         {output_dir}")
    print(f"  Total time:     {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
