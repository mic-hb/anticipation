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
        --test_split_chars f \
        --workers 8

Features:
- Progress bar with tqdm
- Parallel symlink creation
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
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


def create_symlink(args_tuple):
    """Create a single symlink. Called by thread pool."""
    filepath, target_path, use_symlinks = args_tuple

    if target_path.exists():
        return False, "exists"

    try:
        if use_symlinks:
            os.symlink(filepath.resolve(), target_path)
        else:
            shutil.copy2(filepath, target_path)
        return True, "success"
    except Exception as e:
        return False, str(e)


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
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for symlink creation (default: 8)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    start_time = time.time()

    print("=" * 70)
    print("GigaMIDI Restructure to Hex Folders - Parallel")
    print("=" * 70)
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Train:   {args.train_split_chars or '(disabled)'}")
    print(f"Valid:   {args.valid_split_chars or '(disabled)'}")
    print(f"Test:    {args.test_split_chars or '(disabled)'}")
    print(f"Symlinks: {args.use_symlinks}")
    print(f"Workers: {args.workers}")
    print("-" * 70)
    sys.stdout.flush()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        sys.exit(1)

    # Stage 1: Find all MIDI files
    print("\n[Stage 1/4] Scanning for MIDI files...")
    stage_start = time.time()

    mid_files = list(input_dir.glob("*.mid"))
    midi_files = list(input_dir.glob("*.midi"))
    all_files = mid_files + midi_files
    total_files = len(all_files)

    scan_time = time.time() - stage_start
    print(f"  Found {total_files:,} MIDI files in {scan_time:.1f}s")
    sys.stdout.flush()

    # Stage 2: Create directory structure
    print("\n[Stage 2/4] Creating directory structure...")
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

    # Stage 3: Prepare symlink tasks
    print("\n[Stage 3/4] Preparing symlink tasks...")
    stage_start = time.time()

    tasks = []
    train_chars = set(args.train_split_chars)
    valid_chars = set(args.valid_split_chars)
    test_chars = set(args.test_split_chars)

    for filepath in all_files:
        md5 = filepath.stem
        if len(md5) < 1:
            continue

        first_char = md5[0].lower()

        if first_char in train_chars:
            target_base = train_dir
        elif first_char in valid_chars:
            target_base = valid_dir
        elif first_char in test_chars:
            target_base = test_dir
        else:
            continue

        hex_dir = target_base / first_char
        target_path = hex_dir / filepath.name
        tasks.append((filepath, target_path, args.use_symlinks))

    prep_time = time.time() - stage_start
    print(f"  Prepared {len(tasks):,} tasks in {prep_time:.1f}s")
    sys.stdout.flush()

    # Stage 4: Create symlinks with thread pool
    print(f"\n[Stage 4/4] Creating symlinks with {args.workers} workers...")
    stage_start = time.time()
    print("-" * 70)

    train_count = 0
    valid_count = 0
    test_count = 0
    skip_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(create_symlink, t): t for t in tasks}

        with tqdm(total=len(futures), desc="Creating symlinks", unit="files") as pbar:
            for future in as_completed(futures):
                task = futures[future]
                success, status = future.result()

                filepath = task[0]
                md5 = filepath.stem
                first_char = md5[0].lower()

                if success:
                    if first_char in train_chars:
                        train_count += 1
                    elif first_char in valid_chars:
                        valid_count += 1
                    elif first_char in test_chars:
                        test_count += 1
                elif status == "exists":
                    skip_count += 1
                else:
                    error_count += 1

                pbar.update(1)

                if (train_count + valid_count + test_count) % 5000 == 0:
                    elapsed = time.time() - stage_start
                    rate = (
                        (train_count + valid_count + test_count) / elapsed
                        if elapsed > 0
                        else 0
                    )
                    remaining = (
                        len(futures)
                        - train_count
                        - valid_count
                        - test_count
                        - skip_count
                        - error_count
                    )
                    eta = remaining / rate if rate > 0 else 0
                    pbar.set_postfix(
                        {
                            "rate": f"{rate:.0f}f/s",
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
    print(f"  Errors:         {error_count:,}")
    print(f"  Total linked:    {total_linked:,}")
    print(f"  Output:         {output_dir}")
    print(f"  Total time:     {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
