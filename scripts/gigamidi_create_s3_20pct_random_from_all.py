#!/usr/bin/env python3
"""
GigaMIDI Subset S3: 20% Random from Everything

This script creates S3 - a random 20% sample from the ENTIRE GigaMIDI train split.
No filtering applied - truly random sample from all files.

S3 Definition (from docs/amt-fine-tuning.md):
- Subset ID: S3
- Description: 20% random (stratified by instrument group)
- Files: ~280k
- Selection Method: Stratified random sampling by instrument group

Usage:
    python scripts/gigamidi_create_s3_20pct_random_from_all.py [--output data/gigamidi_s3_20pct_random.json]
"""

import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Create S3: 20% random sample from ENTIRE GigaMIDI"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gigamidi_s3_20pct_random.json",
        help="Output JSON file",
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
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=False)

    print("=" * 60)
    print("GigaMIDI Subset S3 Creator")
    print("20% Random from EVERYTHING")
    print("=" * 60)
    print(f"Sample: {args.sample_size * 100}%")
    print(f"Seed: {args.seed}")
    print("-" * 60)
    sys.stdout.flush()

    # Load full dataset
    print("Loading GigaMIDI train split...")
    ds = load_dataset("Metacreation/GigaMIDI", "v2.0.0", split="train", streaming=True)

    # Collect all file metadata
    print("Collecting all file metadata...")
    all_files = []

    pbar = tqdm(desc="Collecting", unit="files", unit_scale=True)
    for row in ds:
        all_files.append(
            {
                "md5": row.get("md5"),
                "num_tracks": row.get("num_tracks", 0),
                "title": row.get("title", ""),
                "artist": row.get("artist", ""),
            }
        )
        pbar.update(1)
        if len(all_files) >= 100000 and len(all_files) % 100000 == 0:
            pbar.set_postfix({"collected": len(all_files)})
    pbar.close()

    total = len(all_files)
    print(f"Total files collected: {total:,}")
    
    # Sort by md5 for reproducibility
    print("Sorting files by md5 for reproducibility...")
    all_files.sort(key=lambda x: x["md5"])
    
    # Random sample
    print(f"Creating {args.sample_size*100}% random sample...")
    random.seed(args.seed)
    sample_size = int(total * args.sample_size)
    s3 = random.sample(all_files, k=sample_size)

    print(f"S3 created: {len(s3):,} files")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(s3, f, indent=2)

    print(f"Saved to: {output_path}")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
