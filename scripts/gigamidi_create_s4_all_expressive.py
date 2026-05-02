#!/usr/bin/env python3
"""
GigaMIDI Subset S4: All Expressive Files

This script creates S4 - ALL files with at least one expressive track (NOMML >= 12).

S4 Definition (from docs/amt-fine-tuning.md):
- Subset ID: S4
- Description: Expressive-all (all files with NOMML >= 12)
- Files: ~859k (31% of dataset)
- Selection Method: NOMML >= 12 filter

Usage:
    python scripts/gigamidi_create_s4_all_expressive.py \
        --output data/gigamidi_s4_all_expressive.json \
        --nomml_threshold 12 \
        --min_tracks 1 \
        --max_tracks 16

Features:
- Progress bar with tqdm
- Time estimates
- NOMML threshold filtering
- Track count filtering
- Summary statistics
"""

import argparse
import json
import sys
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Create S4: All Expressive Files (NOMML >= 12)"
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
        help="Minimum track count (default: 1)",
    )
    parser.add_argument(
        "--max_tracks",
        type=int,
        default=16,
        help="Maximum track count (default: 16)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gigamidi_s4_all_expressive.json",
        help="Output JSON file path",
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
    print("GigaMIDI Subset S4 Creator - All Expressive Files (NOMML >= 12)")
    print("=" * 70)
    print(f"NOMML Threshold: >= {args.nomml_threshold}")
    print(f"Track Range:    {args.min_tracks} - {args.max_tracks}")
    print(f"Output:        {args.output}")
    if args.limit:
        print(f"Limit:         {args.limit:,} files (testing mode)")
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

    # Stage 2: Collect expressive files
    print("\n[Stage 2/4] Filtering for expressive tracks...")
    stage_start = time.time()
    print("-" * 70)
    sys.stdout.flush()

    expressive = []
    total_scanned = 0

    with tqdm(
        desc="Filtering", unit="files", unit_scale=True, unit_divisor=1000
    ) as pbar:
        for i, row in enumerate(ds):
            total_scanned += 1
            pbar.update(1)

            num_tracks = row.get("num_tracks", 0) or 0

            if num_tracks < args.min_tracks or num_tracks > args.max_tracks:
                pbar.update(0)
                continue

            nomml = row.get("NOMML", []) or []
            has_expressive = any(n >= args.nomml_threshold for n in nomml)

            if not has_expressive:
                continue

            expressive.append(
                {
                    "md5": row.get("md5", ""),
                    "nomml": nomml,
                    "num_tracks": num_tracks,
                    "title": row.get("title", ""),
                    "artist": row.get("artist", ""),
                    "duration": row.get("duration", 0),
                    "time_signature": row.get("time_signature", ""),
                    "tempo": row.get("tempo", 0),
                    "styles": row.get("music_styles_curated", []),
                }
            )

            if i > 0 and i % 10000 == 0:
                elapsed = time.time() - stage_start
                rate = i / elapsed if elapsed > 0 else 0
                pbar.set_postfix(
                    {
                        "kept": len(expressive),
                        "rate": f"{rate:.0f}f/s",
                    }
                )

            if args.limit and i >= args.limit - 1:
                break

    scan_time = time.time() - stage_start
    match_rate = 100 * len(expressive) / total_scanned if total_scanned > 0 else 0
    print(
        f"\n  Scanned:      {total_scanned:,} files in {scan_time:.1f}s ({total_scanned / scan_time:.0f} files/s)"
    )
    print(f"  Match rate:   {match_rate:.1f}%")
    sys.stdout.flush()

    # Stage 3: Sort by md5 for reproducibility
    print("\n[Stage 3/4] Sorting by MD5 for reproducibility...")
    stage_start = time.time()
    print("-" * 70)

    total_expressive = len(expressive)
    print(f"  Sorting {total_expressive:,} files by MD5...")
    with tqdm(total=total_expressive, desc="Sorting", unit="files") as pbar:
        expressive.sort(key=lambda x: x["md5"])
        pbar.update(total_expressive)

    sort_time = time.time() - stage_start
    print(f"  Sorted in {sort_time:.1f}s")
    sys.stdout.flush()

    # Stage 4: Save output
    print("\n[Stage 4/4] Saving output...")
    stage_start = time.time()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(expressive, f, indent=2)

    save_time = time.time() - stage_start
    print(f"  Saved in {save_time:.1f}s")
    print(f"  Output: {output_path}")
    sys.stdout.flush()

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Total scanned:     {total_scanned:,}")
    print(f"  S4 expressive:   {total_expressive:,}")
    print(f"  Match rate:      {match_rate:.1f}%")
    print(f"  NOMML threshold: >={args.nomml_threshold}")
    print(f"  Track range:     {args.min_tracks}-{args.max_tracks}")
    print(f"  Output file:     {output_path}")
    print(f"  Total time:     {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
