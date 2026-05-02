#!/usr/bin/env python3
"""
GigaMIDI Subset S3: 20% Random from Everything (Batched Version)

This script creates S3 - a random 20% sample from the ENTIRE GigaMIDI dataset
(train + validation + test splits combined).

This is a BATCHED version that is MUCH FASTER than streaming:
1. Load all metadata at once (non-streaming)
2. Process in parallel using multiple workers
3. This is ~10x faster but uses more memory (~1GB)

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
    python scripts/gigamidi_create_s3_20pct_random_from_all_batched.py \
        --output data/gigamidi_s3_20pct_random_from_all.json \
        --sample_size 0.20 \
        --seed 42

Features:
- FAST BATCHED processing (~10x faster than streaming)
- Progress bar with tqdm
- Parallel chunk processing
- Time estimates
- Deterministic sorting before sampling (reproducibility)
- Samples from ALL splits (train + valid + test)
"""

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def process_chunk(args_tuple):
    """Process a chunk of rows and extract metadata."""
    rows, split_name = args_tuple

    results = []
    for row in rows:
        results.append(
            {
                "md5": row.get("md5", ""),
                "split": split_name,
                "num_tracks": row.get("num_tracks", 0),
                "title": row.get("title", ""),
                "artist": row.get("artist", ""),
                "duration": row.get("duration", 0),
                "time_signature": row.get("time_signature", ""),
                "tempo": row.get("tempo", 0),
            }
        )
    return split_name, results


def main():
    parser = argparse.ArgumentParser(
        description="Create S3: 20% random from ENTIRE GigaMIDI (BATCHED - FAST)"
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
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for chunk processing (default: 8)",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Number of rows per chunk for parallel processing (default: 10000)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    start_time = time.time()

    print("=" * 70)
    print("GigaMIDI Subset S3 Creator - BATCHED (FAST VERSION)")
    print("=" * 70)
    print(f"Sample Size:  {args.sample_size * 100:.1f}%")
    print(f"Seed:       {args.seed}")
    print(f"Output:     {args.output}")
    print(f"Workers:    {args.workers}")
    print(f"Chunk Size: {args.chunk_size:,}")
    if args.limit:
        print(f"Limit:      {args.limit:,} files per split (testing mode)")
    print("-" * 70)
    print("Note: BATCHED - loads all data first, then parallelizes processing")
    print("       This is ~10x faster but uses more memory (~1GB)")
    print("       Hash-based splitting: 0-d -> train, e -> valid, f -> test")
    print("-" * 70)
    sys.stdout.flush()

    # Stage 1: Load ALL data at once (non-streaming)
    print("\n[Stage 1/4] Loading ALL data (non-streaming)...")
    stage_start = time.time()

    all_data = {}
    for split_name in ["train", "validation", "test"]:
        print(f"  Loading {split_name}...")

        if args.limit:
            ds = load_dataset(
                "Metacreation/GigaMIDI",
                "v2.0.0",
                split=split_name,
            )
            ds = ds.select(range(min(args.limit, len(ds))))
        else:
            ds = load_dataset(
                "Metacreation/GigaMIDI",
                "v2.0.0",
                split=split_name,
            )

        all_data[split_name] = list(ds)
        print(f"    Loaded {len(all_data[split_name]):,} rows")

    train_count = len(all_data.get("train", []))
    valid_count = len(all_data.get("validation", []))
    test_count = len(all_data.get("test", []))

    load_time = time.time() - stage_start
    total_rows = train_count + valid_count + test_count
    print(f"  Total loaded: {total_rows:,} rows in {load_time:.1f}s")
    sys.stdout.flush()

    # Stage 2: Parallel processing
    print(f"\n[Stage 2/4] Processing with {args.workers} workers...")
    stage_start = time.time()

    all_files = []
    tasks = []

    for split_name, rows in all_data.items():
        for i in range(0, len(rows), args.chunk_size):
            chunk = rows[i : i + args.chunk_size]
            tasks.append((chunk, split_name))

    print(f"  Created {len(tasks)} chunks of size {args.chunk_size:,}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_chunk, t): t[1] for t in tasks}

        with tqdm(total=len(futures), desc="Processing chunks", unit="chunks") as pbar:
            for future in as_completed(futures):
                name, results = future.result()
                all_files.extend(results)
                pbar.update(1)

    process_time = time.time() - stage_start
    print(f"  Processed {len(all_files):,} files in {process_time:.1f}s")
    sys.stdout.flush()

    # Stage 3: Sort by md5 for reproducibility
    print("\n[Stage 3/4] Sorting by MD5 for reproducibility...")
    stage_start = time.time()

    print(f"  Sorting {len(all_files):,} files by MD5...")
    all_files.sort(key=lambda x: x["md5"])
    print(f"  Sorted in {time.time() - stage_start:.1f}s")
    sys.stdout.flush()

    # Stage 4: Random sampling
    print("\n[Stage 4/4] Random sampling...")
    stage_start = time.time()

    random.seed(args.seed)
    sample_size = int(len(all_files) * args.sample_size)

    print(f"  Total files:  {len(all_files):,}")
    print(f"  Sample size: {args.sample_size * 100:.1f}% ({sample_size:,})")
    print(f"  Seed:        {args.seed}")

    s3_sample = random.sample(all_files, k=sample_size)
    print(f"  Sampled in {time.time() - stage_start:.1f}s")
    print(f"\n  S3 Subset: {len(s3_sample):,} files selected")

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

    # Save output
    print("\n[Saving] Writing output file...")
    stage_start = time.time()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(s3_sample, f, indent=2)

    print(f"  Saved in {time.time() - stage_start:.1f}s")
    print(f"  Output:   {output_path}")

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Total files collected:    {len(all_files):,}")
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
