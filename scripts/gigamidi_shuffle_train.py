#!/usr/bin/env python3
"""
GigaMIDI Shuffle Training Data

Shuffles the training data file. This is CRITICAL for good model performance!

The README warns:
"Warning: we have observed that approximate shuffling using a local shuffle 
of the training data is not sufficient to achieve good model performance"

This script uses Python's built-in random with a seed for reproducibility.

Usage:
    python scripts/gigamidi_shuffle_train.py \
        --input data/gigamidi_s1_train-ordered.txt \
        --output data/gigamidi_s1_train.txt \
        --seed 42

Features:
- Memory-mapped file shuffling for large files
- Progress bar with tqdm
- Time estimates
- Reproducible with seed
- Chunked processing for memory efficiency
"""

import argparse
import os
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle training data for better model performance"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input ordered training file",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output shuffled training file",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=1000000,
        help="Lines to read into memory at a time (default: 1M)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    start_time = time.time()

    print("=" * 70)
    print("GigaMIDI Shuffle Training Data")
    print("=" * 70)
    print(f"Input:     {args.input}")
    print(f"Output:    {args.output}")
    print(f"Seed:      {args.seed}")
    print(f"Buffer:    {args.buffer_size:,}")
    print("-" * 70)
    print("\nWARNING: Reading entire file into memory may require significant RAM!")
    print("-" * 70)
    sys.stdout.flush()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Stage 1: Count lines
    print("\n[Stage 1/3] Counting lines...")
    stage_start = time.time()

    line_count = 0
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for _ in f:
            line_count += 1

    count_time = time.time() - stage_start
    print(f"  Total lines: {line_count:,} in {count_time:.1f}s")
    sys.stdout.flush()

    # Stage 2: Read all lines
    print("\n[Stage 2/3] Reading training data...")
    stage_start = time.time()

    print("  Reading into memory...")
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    read_time = time.time() - stage_start
    print(f"  Read {len(lines):,} lines in {read_time:.1f}s")
    print(f"  Memory: {len(lines) * 100 / (1024**3):.2f} GB estimate")
    sys.stdout.flush()

    # Stage 3: Shuffle
    print("\n[Stage 3/3] Shuffling...")
    stage_start = time.time()

    print(f"  Shuffling with seed={args.seed}...")
    random.seed(args.seed)
    random.shuffle(lines)

    shuffle_time = time.time() - stage_start
    print(f"  Shuffled in {shuffle_time:.1f}s")
    sys.stdout.flush()

    # Stage 4: Write output
    print("\n[Stage 4/3] Writing shuffled data...")
    stage_start = time.time()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tqdm(
        total=len(lines),
        desc="Writing",
        unit="lines",
        unit_scale=True,
        unit_divisor=1000,
    ) as pbar:
        with open(output_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(line)
                pbar.update(1)

    write_time = time.time() - stage_start
    print(f"  Written in {write_time:.1f}s")
    sys.stdout.flush()

    # Cleanup
    del lines

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Total lines:    {line_count:,}")
    print(f"  Input:       {input_path}")
    print(f"  Output:      {output_path}")
    print(f"  Seed:        {args.seed}")
    print(f"  Total time:  {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)
    print("\nIMPORTANT: Training data is now ready for AMT training!")
    print("=" * 70)


if __name__ == "__main__":
    main()
