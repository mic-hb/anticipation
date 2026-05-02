#!/usr/bin/env python3
"""
GigaMIDI Subset S2: 10% Random from Expressive Only (Batched Version)

This script creates S2 - random 10% sample from files that have at least one
expressive track (NOMML >= 12), from the ENTIRE GigaMIDI dataset.

This is a BATCHED version that is MUCH FASTER than streaming:
1. Load all metadata at once (non-streaming)
2. Process in parallel using multiple workers
3. This is ~10x faster but uses more memory (~1GB)

Then, after downloading, use hash-based splitting (Lakh convention):
- md5 starts with 0-d -> train
- md5 starts with e -> validation  
- md5 starts with f -> test

S2 Definition (from docs/amt-fine-tuning.md):
- Subset ID: S2
- Description: 10% random from expressive only (S4)
- Files: ~86k
- Selection Method: NOMML >= 12 filter from all splits, then 10% random

Usage:
    python scripts/gigamidi_create_s2_10pct_random_from_expressive_batched.py \
        --output data/gigamidi_s2_10pct_random_from_expressive.json \
        --sample_size 0.10 \
        --seed 42 \
        --nomml_threshold 12

Features:
- FAST BATCHED processing (~10x faster than streaming)
- Progress bar with tqdm
- Parallel chunk processing
- Time estimates
- NOMML threshold filtering
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


def process_chunk_expressive(args_tuple):
    """Process a chunk of rows and filter for expressive tracks."""
    rows, split_name, nomml_threshold, min_tracks, max_tracks = args_tuple

    results = []
    for row in rows:
        num_tracks = row.get("num_tracks", 0) or 0
        if num_tracks < min_tracks or num_tracks > max_tracks:
            continue

        nomml = row.get("NOMML", []) or []
        has_expressive = any(n >= nomml_threshold for n in nomml)

        if not has_expressive:
            continue

        results.append(
            {
                "md5": row.get("md5", ""),
                "split": split_name,
                "nomml": nomml,
                "num_tracks": num_tracks,
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
        description="Create S2: 10% random from expressive (BATCHED - FAST)"
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
    parser.add_argument(
        "--nomml_threshold",
        type=int,
        default=12,
        help="NOMML threshold for expressive (default: 12)",
    )
    parser.add_argument(
        "--min_tracks",
        type=int,
        default=1,
        help="Minimum number of tracks (default: 1)",
    )
    parser.add_argument(
        "--max_tracks",
        type=int,
        default=16,
        help="Maximum number of tracks (default: 16)",
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
    print("GigaMIDI Subset S2 Creator - BATCHED (FAST VERSION)")
    print("=" * 70)
    print(f"Sample Size:     {args.sample_size * 100:.1f}%")
    print(f"NOMML Threshold: >= {args.nomml_threshold}")
    print(f"Track Range:     {args.min_tracks}-{args.max_tracks}")
    print(f"Seed:           {args.seed}")
    print(f"Output:         {args.output}")
    print(f"Workers:        {args.workers}")
    print(f"Chunk Size:     {args.chunk_size:,}")
    if args.limit:
        print(f"Limit:          {args.limit:,} files per split (testing mode)")
    print("-" * 70)
    print("Note: BATCHED - loads all data first, then parallelizes processing")
    print("       This is ~10x faster but uses more memory (~1GB)")
    print("       Hash-based splitting: 0-d -> train, e -> valid, f -> test")
    print("-" * 70)
    sys.stdout.flush()

    # Stage 1: Load ALL data at once (non-streaming)
    print("\n[Stage 1/5] Loading ALL data (non-streaming)...")
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

    # Stage 2: Parallel processing with filtering
    print(
        f"\n[Stage 2/5] Processing with {args.workers} workers (expressive filter)..."
    )
    stage_start = time.time()

    all_files = []
    tasks = []

    for split_name, rows in all_data.items():
        for i in range(0, len(rows), args.chunk_size):
            chunk = rows[i : i + args.chunk_size]
            tasks.append(
                (
                    chunk,
                    split_name,
                    args.nomml_threshold,
                    args.min_tracks,
                    args.max_tracks,
                )
            )

    print(f"  Created {len(tasks)} chunks of size {args.chunk_size:,}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_chunk_expressive, t): t[1] for t in tasks}

        with tqdm(total=len(futures), desc="Processing chunks", unit="chunks") as pbar:
            for future in as_completed(futures):
                name, results = future.result()
                all_files.extend(results)
                pbar.update(1)

    process_time = time.time() - stage_start

    train_expressive = len([f for f in all_files if f["split"] == "train"])
    valid_expressive = len([f for f in all_files if f["split"] == "validation"])
    test_expressive = len([f for f in all_files if f["split"] == "test"])

    print(f"  Processed {len(all_files):,} expressive files in {process_time:.1f}s")
    print(f"    - Train:      {train_expressive:,}")
    print(f"    - Valid:     {valid_expressive:,}")
    print(f"    - Test:      {test_expressive:,}")
    sys.stdout.flush()

    # Stage 3: Sort by md5 for reproducibility
    print("\n[Stage 3/5] Sorting by MD5 for reproducibility...")
    stage_start = time.time()

    print(f"  Sorting {len(all_files):,} files by MD5...")
    all_files.sort(key=lambda x: x["md5"])
    print(f"  Sorted in {time.time() - stage_start:.1f}s")
    sys.stdout.flush()

    # Stage 4: Random sampling
    print("\n[Stage 4/5] Random sampling...")
    stage_start = time.time()

    random.seed(args.seed)
    sample_size = int(len(all_files) * args.sample_size)

    print(f"  Total expressive: {len(all_files):,}")
    print(f"  Sample size:     {args.sample_size * 100:.1f}% ({sample_size:,})")
    print(f"  Seed:           {args.seed}")

    s2_sample = random.sample(all_files, k=sample_size)
    print(f"  Sampled in {time.time() - stage_start:.1f}s")
    print(f"\n  S2 Subset: {len(s2_sample):,} files selected")

    # Analyze resulting splits based on hash
    train_hashes = set("0123456789abcd")
    valid_hashes = set("e")
    test_hashes = set("f")

    train_result = sum(1 for f in s2_sample if f["md5"][0].lower() in train_hashes)
    valid_result = sum(1 for f in s2_sample if f["md5"][0].lower() in valid_hashes)
    test_result = sum(1 for f in s2_sample if f["md5"][0].lower() in test_hashes)

    print(f"\n  After hash-based splitting:")
    print(
        f"    - Train (0-d): {train_result:,} ({100 * train_result / len(s2_sample):.1f}%)"
    )
    print(
        f"    - Valid (e):   {valid_result:,} ({100 * valid_result / len(s2_sample):.1f}%)"
    )
    print(
        f"    - Test (f):    {test_result:,} ({100 * test_result / len(s2_sample):.1f}%)"
    )

    # Stage 5: Save output
    print("\n[Stage 5/5] Writing output file...")
    stage_start = time.time()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(s2_sample, f, indent=2)

    print(f"  Saved in {time.time() - stage_start:.1f}s")
    print(f"  Output:   {output_path}")

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Total files collected:    {total_rows:,}")
    print(f"    - Train:              {train_count:,}")
    print(f"    - Valid:             {valid_count:,}")
    print(f"    - Test:               {test_count:,}")
    print(f"  Expressive files:        {len(all_files):,}")
    print(f"    - Train:              {train_expressive:,}")
    print(f"    - Valid:             {valid_expressive:,}")
    print(f"    - Test:               {test_expressive:,}")
    print(f"  S2 subset size:          {len(s2_sample):,}")
    print(f"  Sample percentage:      {args.sample_size * 100:.1f}%")
    print(f"  Output file:            {output_path}")
    print(f"  Total time:            {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)
    print("\nNote: After download, use hash-based splitting:")
    print("       0-d -> train/, e -> valid/, f -> test/")


if __name__ == "__main__":
    main()
