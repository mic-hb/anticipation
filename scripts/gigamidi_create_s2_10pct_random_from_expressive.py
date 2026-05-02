#!/usr/bin/env python3
"""
GigaMIDI Subset S2: 10% Random from Expressive Only

This script creates S2 - random 10% sample from files that have at least one
expressive track (NOMML >= 12).

S2 Definition (from docs/amt-fine-tuning.md):
- Subset ID: S2
- Description: 10% random from expressive only (S4)
- Files: ~86k
- Selection Method: NOMML >= 12 filter, then 10% random sample

Usage:
    python scripts/gigamidi_create_s2_10pct_random_from_expressive.py \
        --s4_input data/gigamidi_s4_all_expressive.json \
        --output data/gigamidi_s2_10pct_random_from_expressive.json \
        --sample_size 0.10 \
        --seed 42

Features:
- Progress bar with tqdm
- Time estimates
- Deterministic sorting before sampling (reproducibility)
- Summary statistics
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Create S2: 10% random sample from expressive files (S4)"
    )
    parser.add_argument(
        "--s4_input",
        type=str,
        default="data/gigamidi_s4_all_expressive.json",
        help="Input S4 file (from gigamidi_create_s4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gigamidi_s2_10pct_random_from_expressive.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--sample_size",
        type=float,
        default=0.10,
        help="Sample percentage (default: 0.10 = 10%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    start_time = time.time()

    print("=" * 70)
    print("GigaMIDI Subset S2 Creator - 10% Random from Expressive")
    print("=" * 70)
    print(f"S4 Input: {args.s4_input}")
    print(f"Output:  {args.output}")
    print(f"Sample:  {args.sample_size * 100:.1f}%")
    print(f"Seed:    {args.seed}")
    print("-" * 70)
    sys.stdout.flush()

    # Stage 1: Load S4
    print("\n[Stage 1/4] Loading S4 (expressive files)...")
    stage_start = time.time()

    with open(args.s4_input) as f:
        s4 = json.load(f)

    total_s4 = len(s4)
    load_time = time.time() - stage_start
    print(f"  Loaded {total_s4:,} expressive files in {load_time:.1f}s")
    sys.stdout.flush()

    # Stage 2: Sort by md5 for reproducibility
    print("\n[Stage 2/4] Sorting by MD5 for reproducibility...")
    stage_start = time.time()
    print("-" * 70)

    print(f"  Sorting {total_s4:,} files by MD5...")
    with tqdm(total=total_s4, desc="Sorting", unit="files") as pbar:
        s4.sort(key=lambda x: x["md5"])
        pbar.update(total_s4)

    sort_time = time.time() - stage_start
    print(f"  Sorted in {sort_time:.1f}s")
    sys.stdout.flush()

    # Stage 3: Random sampling
    print("\n[Stage 3/4] Random sampling...")
    stage_start = time.time()
    print("-" * 70)

    random.seed(args.seed)
    sample_size = int(total_s4 * args.sample_size)

    print(f"  S4 files:      {total_s4:,}")
    print(f"  Sample size:  {sample_size:,} ({args.sample_size * 100:.1f}%)")
    print(f"  Seed:         {args.seed}")
    sys.stdout.flush()

    s2_sample = random.sample(s4, k=sample_size)

    sample_time = time.time() - stage_start
    print(f"  Sampled in {sample_time:.1f}s")
    print(f"\n  S2 Subset: {len(s2_sample):,} files selected")
    sys.stdout.flush()

    # Stage 4: Save output
    print("\n[Stage 4/4] Saving output...")
    stage_start = time.time()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(s2_sample, f, indent=2)

    save_time = time.time() - stage_start
    print(f"  Saved in {save_time:.1f}s")
    print(f"  Output: {output_path}")
    sys.stdout.flush()

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  S4 (expressive):     {total_s4:,}")
    print(f"  S2 subset size:      {len(s2_sample):,}")
    print(f"  Sample percentage:  {args.sample_size * 100:.1f}%")
    print(f"  Output file:        {output_path}")
    print(f"  Total time:         {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
