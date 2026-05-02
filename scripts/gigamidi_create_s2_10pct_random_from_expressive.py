#!/usr/bin/env python3
"""
GigaMIDI Subset S2: 10% Random from Expressive Only

This script creates S2 - random 10% sample from files that have at least one
expressive track (NOMML >= 12).

S2 Definition (from docs/amt-fine-tuning.md):
- Subset ID: S2
- Description: 10% expressive-only
- Files: ~44k
- Selection Method: NOMML >= 12 filter, then 10% random sample

Usage:
    python scripts/gigamidi_create_s2.py [--s4_input data/gigamidi_s4_expressive.json]
                               [--output data/gigamidi_s2_expressive_10pct.json]
                               [--sample_size 0.10]
"""

import argparse
import json
import random
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Create S2: 10% Random from Expressive Files"
    )
    parser.add_argument(
        "--s4_input",
        type=str,
        default="data/gigamidi_s4_all_expressive.json",
        help="Input S4 file (from gigamidi_create_s4.py)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gigamidi_s2_10pct_random_from_expressive.json",
        help="Output JSON file",
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
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=False)

    print("=" * 60)
    print("GigaMIDI Subset S2 Creator")
    print("10% Random from Expressive Files")
    print("=" * 60)
    print(f"Input: {args.s4_input}")
    print(f"Sample: {args.sample_size * 100}%")
    print(f"Seed: {args.seed}")
    print("-" * 60)

    # Load S4 (all expressive files)
    print("Loading S4 (expressive files)...")
    with open(args.s4_input) as f:
        s4 = json.load(f)

    print(f"S4 files: {len(s4):,}")

    # Sort by md5 for reproducibility
    print("Sorting S4 files by md5...")
    s4.sort(key=lambda x: x["md5"])

    # Random sample from S4
    random.seed(args.seed)
    sample_size = int(len(s4) * args.sample_size)
    s2 = random.sample(s4, k=sample_size)

    print(f"S2 created: {len(s2):,} files")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(s2, f, indent=2)

    print(f"Saved to: {output_path}")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
