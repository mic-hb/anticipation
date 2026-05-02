#!/usr/bin/env python3
"""
GigaMIDI Subset S3: 20% Random from Everything

This script creates S3 - a random 20% sample from the ENTIRE GigaMIDI train split.
No filtering applied - truly random sample from all files.

S3 Definition (from docs/amt-fine-tuning.md):
- Subset ID: S3
- Description: 20% random from ALL
- Files: ~209k
- Selection Method: Stratified random sampling by instrument group

Usage:
    python scripts/gigamidi_create_s3_20pct_random_from_all.py \
        --output data/gigamidi_s3_20pct_random_from_all.json \
        --sample_size 0.20 \
        --seed 42

Features:
- Progress bar with tqdm
- Time estimates
- Memory-efficient streaming
- Deterministic sorting before sampling (reproducibility)
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Create S3: 20% random sample from ENTIRE GigaMIDI train split"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gigamidi_s3_20pct_random_from_all.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--sample_size",
        type=float,
        default=0.20,
        help="Sample percentage (default: 0.20 = 20%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    start_time = time.time()

    print("=" * 70)
    print("GigaMIDI Subset S3 Creator - 20% Random from ENTIRE Dataset")
    print("=" * 70)
    print(f"Sample Size:  {args.sample_size * 100:.1f}%")
    print(f"Seed:       {args.seed}")
    print(f"Output:     {args.output}")
    if args.limit:
        print(f"Limit:      {args.limit:,} files (testing mode)")
    print("-" * 70)
    sys.stdout.flush()

    # Stage 1: Load dataset
    print("\n[Stage 1/4] Loading GigaMIDI train split...")
    stage_start = time.time()

    ds = load_dataset(
        "Metacreation/GigaMIDI",
        "v2.0.0",
        split="train",
        streaming=True,
    )

    # Stage 2: Collect file metadata
    print("\n[Stage 2/4] Collecting file metadata...")
    stage_start = time.time()
    print("-" * 70)
    sys.stdout.flush()

    all_files = []

    with tqdm(
        desc="Collecting files", unit="files", unit_scale=True, unit_divisor=1000
    ) as pbar:
        for i, row in enumerate(ds):
            all_files.append(
                {
                    "md5": row.get("md5", ""),
                    "num_tracks": row.get("num_tracks", 0),
                    "title": row.get("title", ""),
                    "artist": row.get("artist", ""),
                    "duration": row.get("duration", 0),
                    "time_signature": row.get("time_signature", ""),
                    "tempo": row.get("tempo", 0),
                }
            )
            pbar.update(1)

            if args.limit and i >= args.limit - 1:
                break

            if i > 0 and i % 10000 == 0:
                elapsed = time.time() - stage_start
                rate = i / elapsed
                pbar.set_postfix({"files/s": f"{rate:.0f}"})

    total_files = len(all_files)
    stage_time = time.time() - stage_start
    print(
        f"\n  Collected: {total_files:,} files in {stage_time:.1f}s ({total_files / stage_time:.0f} files/s)"
    )
    sys.stdout.flush()

    # Stage 3: Sort by md5 for reproducibility
    print("\n[Stage 3/4] Sorting by MD5 for reproducibility...")
    stage_start = time.time()
    print("-" * 70)
    sys.stdout.flush()

    print(f"  Sorting {total_files:,} files by MD5...")
    with tqdm(total=total_files, desc="Sorting", unit="files") as pbar:
        all_files.sort(key=lambda x: x["md5"])
        pbar.update(total_files)

    sort_time = time.time() - stage_start
    print(f"  Sorted in {sort_time:.1f}s")
    sys.stdout.flush()

    # Stage 4: Random sampling
    print("\n[Stage 4/4] Random sampling...")
    stage_start = time.time()
    print("-" * 70)

    random.seed(args.seed)
    sample_size = int(total_files * args.sample_size)

    print(f"  Total files:   {total_files:,}")
    print(f"  Sample size:  {sample_size:,} ({args.sample_size * 100:.1f}%)")
    print(f"  Seed:         {args.seed}")
    sys.stdout.flush()

    s3_sample = random.sample(all_files, k=sample_size)

    sample_time = time.time() - stage_start
    print(f"  Sampled in {sample_time:.1f}s")
    print(f"\n  S3 Subset: {len(s3_sample):,} files selected")
    sys.stdout.flush()

    # Save output
    print("\n[Saving] Writing output file...")
    stage_start = time.time()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(s3_sample, f, indent=2)

    save_time = time.time() - stage_start
    print(f"  Saved in {save_time:.1f}s")
    print(f"  Output: {output_path}")
    sys.stdout.flush()

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Total files collected:  {total_files:,}")
    print(f"  S3 subset size:          {len(s3_sample):,}")
    print(f"  Sample percentage:     {args.sample_size * 100:.1f}%")
    print(f"  Output file:          {output_path}")
    print(f"  Total time:          {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
