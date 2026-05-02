#!/usr/bin/env python3
"""
GigaMIDI Subset S3: 20% Random from Everything

This script creates S3 - a random 20% sample from the ENTIRE GigaMIDI dataset
(train + validation + test splits combined).

Then, after downloading, use hash-based splitting (Lakh convention):
- md5 starts with 0-d -> train
- md5 starts with e -> validation  
- md5 starts with f -> test

S3 Definition (from docs/amt-fine-tuning.md):
- Subset ID: S3
- Description: 20% random from ALL
- Files: ~209k
- Selection Method: Random sampling from all splits, then hash-based train/valid/test split

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
- Samples from ALL splits (train + valid + test)
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
        description="Create S3: 20% random sample from ENTIRE GigaMIDI (all splits)"
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
        help="Limit number of files per split to process (for testing)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    start_time = time.time()

    print("=" * 70)
    print("GigaMIDI Subset S3 Creator - 20% Random from EVERYTHING")
    print("=" * 70)
    print(f"Sample Size:  {args.sample_size * 100:.1f}%")
    print(f"Seed:       {args.seed}")
    print(f"Output:     {args.output}")
    if args.limit:
        print(f"Limit:      {args.limit:,} files per split (testing mode)")
    print("-" * 70)
    print("Note: Samples from ALL splits, then uses hash-based splitting:")
    print("       0-d -> train, e -> valid, f -> test")
    print("-" * 70)
    sys.stdout.flush()

    all_files = []

    # Stage 1: Collect from train
    print("\n[Stage 1/4] Collecting from train split...")
    stage_start = time.time()

    ds_train = load_dataset(
        "Metacreation/GigaMIDI",
        "v2.0.0",
        split="train",
        streaming=True,
    )

    with tqdm(
        desc="Train split", unit="files", unit_scale=True, unit_divisor=1000
    ) as pbar:
        for row in ds_train:
            all_files.append(
                {
                    "md5": row.get("md5", ""),
                    "split": "train",
                    "num_tracks": row.get("num_tracks", 0),
                    "title": row.get("title", ""),
                    "artist": row.get("artist", ""),
                    "duration": row.get("duration", 0),
                    "time_signature": row.get("time_signature", ""),
                    "tempo": row.get("tempo", 0),
                }
            )
            pbar.update(1)
            if args.limit and pbar.n >= args.limit:
                break

    train_count = pbar.n
    print(f"  Collected {train_count:,} from train")
    sys.stdout.flush()

    # Stage 2: Collect from validation
    print("\n[Stage 2/4] Collecting from validation split...")
    stage_start = time.time()

    ds_valid = load_dataset(
        "Metacreation/GigaMIDI",
        "v2.0.0",
        split="validation",
        streaming=True,
    )

    with tqdm(
        desc="Valid split", unit="files", unit_scale=True, unit_divisor=1000
    ) as pbar:
        for row in ds_valid:
            all_files.append(
                {
                    "md5": row.get("md5", ""),
                    "split": "validation",
                    "num_tracks": row.get("num_tracks", 0),
                    "title": row.get("title", ""),
                    "artist": row.get("artist", ""),
                    "duration": row.get("duration", 0),
                    "time_signature": row.get("time_signature", ""),
                    "tempo": row.get("tempo", 0),
                }
            )
            pbar.update(1)
            if args.limit and pbar.n >= args.limit:
                break

    valid_count = pbar.n
    print(f"  Collected {valid_count:,} from validation")
    sys.stdout.flush()

    # Stage 3: Collect from test
    print("\n[Stage 3/4] Collecting from test split...")
    stage_start = time.time()

    ds_test = load_dataset(
        "Metacreation/GigaMIDI",
        "v2.0.0",
        split="test",
        streaming=True,
    )

    with tqdm(
        desc="Test split", unit="files", unit_scale=True, unit_divisor=1000
    ) as pbar:
        for row in ds_test:
            all_files.append(
                {
                    "md5": row.get("md5", ""),
                    "split": "test",
                    "num_tracks": row.get("num_tracks", 0),
                    "title": row.get("title", ""),
                    "artist": row.get("artist", ""),
                    "duration": row.get("duration", 0),
                    "time_signature": row.get("time_signature", ""),
                    "tempo": row.get("tempo", 0),
                }
            )
            pbar.update(1)
            if args.limit and pbar.n >= args.limit:
                break

    test_count = pbar.n
    print(f"  Collected {test_count:,} from test")
    sys.stdout.flush()

    total_files = len(all_files)
    print(f"\n  Total collected: {total_files:,}")
    print(f"    - Train:      {train_count:,} ({100 * train_count / total_files:.1f}%)")
    print(f"    - Valid:     {valid_count:,} ({100 * valid_count / total_files:.1f}%)")
    print(f"    - Test:      {test_count:,} ({100 * test_count / total_files:.1f}%)")
    sys.stdout.flush()

    # Stage 4: Sort by md5 for reproducibility
    print("\n[Stage 4/4] Sorting by MD5 for reproducibility...")
    stage_start = time.time()
    print("-" * 70)

    print(f"  Sorting {total_files:,} files by MD5...")
    with tqdm(total=total_files, desc="Sorting", unit="files") as pbar:
        all_files.sort(key=lambda x: x["md5"])
        pbar.update(total_files)

    sort_time = time.time() - stage_start
    print(f"  Sorted in {sort_time:.1f}s")
    sys.stdout.flush()

    # Stage 5: Random sampling
    print("\n[Stage 5/5] Random sampling...")
    stage_start = time.time()
    print("-" * 70)

    random.seed(args.seed)
    sample_size = int(total_files * args.sample_size)

    print(f"  Total files:  {total_files:,}")
    print(f"  Sample size: {args.sample_size * 100:.1f}% ({sample_size:,})")
    print(f"  Seed:        {args.seed}")
    sys.stdout.flush()

    s3_sample = random.sample(all_files, k=sample_size)

    sample_time = time.time() - stage_start
    print(f"  Sampled in {sample_time:.1f}s")
    print(f"\n  S3 Subset: {len(s3_sample):,} files selected")
    sys.stdout.flush()

    # Analyze resulting splits based on hash
    train_hashes = set("0123456789abcd")
    valid_hashes = set("e")
    test_hashes = set("f")

    train_result = sum(1 for f in s3_sample if f["md5"][0].lower() in train_hashes)
    valid_result = sum(1 for f in s3_sample if f["md5"][0].lower() in valid_hashes)
    test_result = sum(1 for f in s3_sample if f["md5"][0].lower() in test_hashes)

    print(f"\n  After hash-based splitting:")
    print(
        f"    - Train (0-d): {train_result:,} ({100 * train_result / len(s3_sample):.1f}%)"
    )
    print(
        f"    - Valid (e):   {valid_result:,} ({100 * valid_result / len(s3_sample):.1f}%)"
    )
    print(
        f"    - Test (f):    {test_result:,} ({100 * test_result / len(s3_sample):.1f}%)"
    )
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
    print(f"  Total files collected:    {total_files:,}")
    print(f"    - Train:              {train_count:,}")
    print(f"    - Valid:             {valid_count:,}")
    print(f"    - Test:               {test_count:,}")
    print(f"  S3 subset size:           {len(s3_sample):,}")
    print(f"  Sample percentage:       {args.sample_size * 100:.1f}%")
    print(f"  Output file:             {output_path}")
    print(f"  Total time:             {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)
    print("\nNote: After download, use hash-based splitting:")
    print("       0-d -> train/, e -> valid/, f -> test/")


if __name__ == "__main__":
    main()
