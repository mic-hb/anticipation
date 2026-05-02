#!/usr/bin/env python3
"""
GigaMIDI Subset S2: 10% Random from Expressive Only

This script creates S2 - random 10% sample from files that have at least one
expressive track (NOMML >= 12), from the ENTIRE GigaMIDI dataset.

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
    python scripts/gigamidi_create_s2_10pct_random_from_expressive.py \
        --output data/gigamidi_s2_10pct_random_from_expressive.json \
        --sample_size 0.10 \
        --seed 42

Features:
- Progress bar with tqdm
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
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Create S2: 10% random sample from expressive files (all splits)"
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
        "--limit",
        type=int,
        default=None,
        help="Limit number of files per split to process (for testing)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    start_time = time.time()

    print("=" * 70)
    print("GigaMIDI Subset S2 Creator - 10% Random from Expressive (ALL Splits)")
    print("=" * 70)
    print(f"Sample Size:    {args.sample_size * 100:.1f}%")
    print(f"NOMML Threshold: >= {args.nomml_threshold}")
    print(f"Seed:         {args.seed}")
    print(f"Output:       {args.output}")
    if args.limit:
        print(f"Limit:        {args.limit:,} files per split (testing mode)")
    print("-" * 70)
    print("Note: Samples from ALL splits, then uses hash-based splitting:")
    print("       0-d -> train, e -> valid, f -> test")
    print("-" * 70)
    sys.stdout.flush()

    all_files = []

    for split_name in ["train", "validation", "test"]:
        print(f"\n[Collecting] {split_name} split...")

        ds = load_dataset(
            "Metacreation/GigaMIDI",
            "v2.0.0",
            split=split_name,
            streaming=True,
        )

        split_count = 0
        expressive_count = 0

        with tqdm(
            desc=f"{split_name}", unit="files", unit_scale=True, unit_divisor=1000
        ) as pbar:
            for row in ds:
                split_count += 1
                pbar.update(1)

                num_tracks = row.get("num_tracks", 0) or 0
                if num_tracks < 1 or num_tracks > 16:
                    continue

                nomml = row.get("NOMML", []) or []
                has_expressive = any(n >= args.nomml_threshold for n in nomml)

                if not has_expressive:
                    continue

                all_files.append(
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
                expressive_count += 1

                if args.limit and expressive_count >= args.limit:
                    break

                if split_count % 10000 == 0:
                    pbar.set_postfix({"expressive": expressive_count})

        print(
            f"  {split_name}: {split_count:,} scanned, {expressive_count:,} expressive"
        )

    total_files = len(all_files)
    print(f"\n  Total expressive files: {total_files:,}")
    sys.stdout.flush()

    # Sort by md5 for reproducibility
    print("\n[Sorting] by MD5 for reproducibility...")
    all_files.sort(key=lambda x: x["md5"])
    print(f"  Sorted {total_files:,} files")

    # Random sampling
    print("\n[Sampling] Random sampling...")
    random.seed(args.seed)
    sample_size = int(total_files * args.sample_size)

    print(f"  Total expressive: {total_files:,}")
    print(f"  Sample size: {sample_size:,} ({args.sample_size * 100:.1f}%)")
    print(f"  Seed: {args.seed}")

    s2_sample = random.sample(all_files, k=sample_size)
    print(f"  Sampled: {len(s2_sample):,} files")

    # Analyze resulting splits based on hash
    train_hashes = set("0123456789abcd")
    valid_hashes = set("e")
    test_hashes = set("f")

    train_result = sum(1 for f in s2_sample if f["md5"][0].lower() in train_hashes)
    valid_result = sum(1 for f in s2_sample if f["md5"][0].lower() in valid_hashes)
    test_result = sum(1 for f in s2_sample if f["md5"][0].lower() in test_hashes)

    print(f"\n  After hash-based splitting:")
    print(f"    - Train (0-d): {train_result:,}")
    print(f"    - Valid (e):   {valid_result:,}")
    print(f"    - Test (f):    {test_result:,}")

    # Save output
    print("\n[Saving] Writing output file...")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(s2_sample, f, indent=2)

    print(f"  Saved to: {output_path}")

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Total expressive files:  {total_files:,}")
    print(f"  S2 subset size:         {len(s2_sample):,}")
    print(f"  Sample percentage:     {args.sample_size * 100:.1f}%")
    print(f"  Output file:           {output_path}")
    print(f"  Total time:           {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
