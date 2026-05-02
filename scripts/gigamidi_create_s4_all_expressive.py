#!/usr/bin/env python3
"""
GigaMIDI Subset S4: All Expressive Files

This script creates S4 - ALL files with at least one expressive track (NOMML >= 12)
from the ENTIRE GigaMIDI dataset (train + validation + test splits combined).

Then, after downloading, use hash-based splitting (Lakh convention):
- md5 starts with 0-d -> train
- md5 starts with e -> validation  
- md5 starts with f -> test

S4 Definition (from docs/amt-fine-tuning.md):
- Subset ID: S4
- Description: Expressive-all (all files with NOMML >= 12)
- Files: ~859k (31% of dataset)
- Selection Method: NOMML >= 12 filter from all splits

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
- Samples from ALL splits (train + valid + test)
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
        description="Create S4: All Expressive Files (NOMML >= 12) from ALL splits"
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
        help="Limit number of files per split to process (for testing)",
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
        print(f"Limit:         {args.limit:,} files per split (testing mode)")
    print("-" * 70)
    print("Note: Collects from ALL splits, then uses hash-based splitting:")
    print("       0-d -> train, e -> valid, f -> test")
    print("-" * 70)
    sys.stdout.flush()

    all_expressive = []

    # Stage 1: Collect from train split
    print("\n[Stage 1/4] Collecting from train split...")
    stage_start = time.time()

    ds_train = load_dataset(
        "Metacreation/GigaMIDI",
        "v2.0.0",
        split="train",
        streaming=True,
    )

    train_scanned = 0
    train_expressive = 0

    with tqdm(
        desc="Train split", unit="files", unit_scale=True, unit_divisor=1000
    ) as pbar:
        for row in ds_train:
            train_scanned += 1
            pbar.update(1)

            num_tracks = row.get("num_tracks", 0) or 0

            if num_tracks < args.min_tracks or num_tracks > args.max_tracks:
                continue

            nomml = row.get("NOMML", []) or []
            has_expressive = any(n >= args.nomml_threshold for n in nomml)

            if not has_expressive:
                continue

            train_expressive += 1
            all_expressive.append(
                {
                    "md5": row.get("md5", ""),
                    "split": "train",
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

            if args.limit and train_expressive >= args.limit:
                break

            if train_scanned > 0 and train_scanned % 10000 == 0:
                elapsed = time.time() - stage_start
                rate = train_scanned / elapsed if elapsed > 0 else 0
                pbar.set_postfix(
                    {
                        "kept": train_expressive,
                        "rate": f"{rate:.0f}f/s",
                    }
                )

    print(f"  Scanned: {train_scanned:,}, Expressive: {train_expressive:,}")
    sys.stdout.flush()

    # Stage 2: Collect from validation split
    print("\n[Stage 2/4] Collecting from validation split...")
    stage_start = time.time()

    ds_valid = load_dataset(
        "Metacreation/GigaMIDI",
        "v2.0.0",
        split="validation",
        streaming=True,
    )

    valid_scanned = 0
    valid_expressive = 0

    with tqdm(
        desc="Valid split", unit="files", unit_scale=True, unit_divisor=1000
    ) as pbar:
        for row in ds_valid:
            valid_scanned += 1
            pbar.update(1)

            num_tracks = row.get("num_tracks", 0) or 0

            if num_tracks < args.min_tracks or num_tracks > args.max_tracks:
                continue

            nomml = row.get("NOMML", []) or []
            has_expressive = any(n >= args.nomml_threshold for n in nomml)

            if not has_expressive:
                continue

            valid_expressive += 1
            all_expressive.append(
                {
                    "md5": row.get("md5", ""),
                    "split": "validation",
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

            if args.limit and valid_expressive >= args.limit:
                break

    print(f"  Scanned: {valid_scanned:,}, Expressive: {valid_expressive:,}")
    sys.stdout.flush()

    # Stage 3: Collect from test split
    print("\n[Stage 3/4] Collecting from test split...")
    stage_start = time.time()

    ds_test = load_dataset(
        "Metacreation/GigaMIDI",
        "v2.0.0",
        split="test",
        streaming=True,
    )

    test_scanned = 0
    test_expressive = 0

    with tqdm(
        desc="Test split", unit="files", unit_scale=True, unit_divisor=1000
    ) as pbar:
        for row in ds_test:
            test_scanned += 1
            pbar.update(1)

            num_tracks = row.get("num_tracks", 0) or 0

            if num_tracks < args.min_tracks or num_tracks > args.max_tracks:
                continue

            nomml = row.get("NOMML", []) or []
            has_expressive = any(n >= args.nomml_threshold for n in nomml)

            if not has_expressive:
                continue

            test_expressive += 1
            all_expressive.append(
                {
                    "md5": row.get("md5", ""),
                    "split": "test",
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

            if args.limit and test_expressive >= args.limit:
                break

    print(f"  Scanned: {test_scanned:,}, Expressive: {test_expressive:,}")
    sys.stdout.flush()

    total_scanned = train_scanned + valid_scanned + test_scanned
    total_expressive = len(all_expressive)
    match_rate = 100 * total_expressive / total_scanned if total_scanned > 0 else 0

    print(f"\n  Total scanned:   {total_scanned:,}")
    print(f"  Total expressive: {total_expressive:,}")
    print(f"  Match rate:       {match_rate:.1f}%")
    sys.stdout.flush()

    # Stage 4: Sort by md5 for reproducibility
    print("\n[Stage 4/4] Sorting by MD5 for reproducibility...")
    stage_start = time.time()
    print("-" * 70)

    print(f"  Sorting {total_expressive:,} files by MD5...")
    with tqdm(total=total_expressive, desc="Sorting", unit="files") as pbar:
        all_expressive.sort(key=lambda x: x["md5"])
        pbar.update(total_expressive)

    sort_time = time.time() - stage_start
    print(f"  Sorted in {sort_time:.1f}s")
    sys.stdout.flush()

    # Analyze resulting splits based on hash
    train_hashes = set("0123456789abcd")
    valid_hashes = set("e")
    test_hashes = set("f")

    train_result = sum(1 for f in all_expressive if f["md5"][0].lower() in train_hashes)
    valid_result = sum(1 for f in all_expressive if f["md5"][0].lower() in valid_hashes)
    test_result = sum(1 for f in all_expressive if f["md5"][0].lower() in test_hashes)

    print(f"\n  After hash-based splitting:")
    print(
        f"    - Train (0-d): {train_result:,} ({100 * train_result / total_expressive:.1f}%)"
    )
    print(
        f"    - Valid (e):   {valid_result:,} ({100 * valid_result / total_expressive:.1f}%)"
    )
    print(
        f"    - Test (f):    {test_result:,} ({100 * test_result / total_expressive:.1f}%)"
    )
    sys.stdout.flush()

    # Stage 5: Save output
    print("\n[Stage 5/5] Saving output...")
    stage_start = time.time()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_expressive, f, indent=2)

    save_time = time.time() - stage_start
    print(f"  Saved in {save_time:.1f}s")
    print(f"  Output: {output_path}")
    sys.stdout.flush()

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Total scanned:      {total_scanned:,}")
    print(f"    - Train:         {train_scanned:,}")
    print(f"    - Valid:         {valid_scanned:,}")
    print(f"    - Test:          {test_scanned:,}")
    print(f"  S4 expressive:        {total_expressive:,}")
    print(f"  Match rate:         {match_rate:.1f}%")
    print(f"  NOMML threshold:    >={args.nomml_threshold}")
    print(f"  Track range:        {args.min_tracks}-{args.max_tracks}")
    print(f"  Output file:        {output_path}")
    print(f"  Total time:        {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)
    print("\nNote: After download, use hash-based splitting:")
    print("       0-d -> train/, e -> valid/, f -> test/")


if __name__ == "__main__":
    main()
