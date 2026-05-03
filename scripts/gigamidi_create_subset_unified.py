#!/usr/bin/env python3
"""
GigaMIDI Unified Subset Creator (Direct from Cache)

This script creates GigaMIDI subsets (S1, S2, S3, S4) by loading the ENTIRE
dataset at once (including MIDI binaries), filtering/sampling, and writing directly
to the CORRECT folder structure (train/valid/test by hash).

No streaming - single load, filter, write!

Folder Structure (Lakh convention):
    output/
    ├── train/
    │   ├── 0/  (md5 starts with 0-9, a-d)
    │   ├── 1/
    │   ...
    │   ├── d/
    ├── valid/
    │   └── e/  (md5 starts with e)
    └── test/
        └── f/  (md5 starts with f)

Usage:
    # S1: 10% random from all
    python gigamidi_create_subset_unified.py \
        --subset s1 \
        --sample_size 0.10 \
        --seed 42 \
        --output data/gigamidi_s1_10pct_random_from_all/ \
        --workers 8

    # S2: 10% random from expressive
    python gigamidi_create_subset_unified.py \
        --subset s2 \
        --sample_size 0.10 \
        --seed 42 \
        --nomml_threshold 12 \
        --output data/gigamidi_s2_10pct_random_from_expressive/ \
        --workers 8

    # S3: 20% random from all
    python gigamidi_create_subset_unified.py \
        --subset s3 \
        --sample_size 0.20 \
        --seed 42 \
        --output data/gigamidi_s3_20pct_random_from_all/ \
        --workers 8

    # S4: all expressive
    python gigamidi_create_subset_unified.py \
        --subset s4 \
        --nomml_threshold 12 \
        --output data/gigamidi_s4_all_expressive/ \
        --workers 8

Features:
- Loads entire dataset once (including MIDI binary)
- Single-pass filtering
- Parallel MIDI file writing
- Full progress tracking with tqdm
- Hash-based train/valid/test split (Lakh convention)
- Verbose output
"""

import argparse
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


TRAIN_HASHES = set("0123456789abcd")
VALID_HASHES = set("e")
TEST_HASHES = set("f")


def get_split_from_hash(md5: str) -> str:
    """Determine split based on leading hash character (Lakh convention)."""
    first_char = md5[0].lower()
    if first_char in TRAIN_HASHES:
        return "train"
    elif first_char in VALID_HASHES:
        return "valid"
    elif first_char in TEST_HASHES:
        return "test"
    return "train"


def get_hex_folder(md5: str) -> str:
    """Get hex folder (0-f) based on leading hash character."""
    return md5[0].lower()


def write_midi_file(args_tuple):
    """Write a single MIDI file to hex folder (flat structure like Lakh)."""
    midi_bytes, md5, output_base = args_tuple

    split = get_split_from_hash(md5)
    # Flat structure: md5 first char determines hex folder
    hex_folder = get_hex_folder(md5)

    # Write directly to hex folder (not nested under train/valid/test)
    folder = output_base / hex_folder
    folder.mkdir(parents=True, exist_ok=True)

    midi_path = folder / f"{md5}.mid"
    with open(midi_path, "wb") as f:
        f.write(midi_bytes)

    return md5, split


def get_subset_description(subset: str, sample_size: float) -> str:
    """Get human-readable subset description."""
    descriptions = {
        "s1": f"{int(sample_size * 100)}% random from ALL",
        "s2": f"{int(sample_size * 100)}% random from expressive",
        "s3": f"{int(sample_size * 100)}% random from ALL",
        "s4": "ALL expressive files",
    }
    return descriptions.get(subset, subset)


def main():
    parser = argparse.ArgumentParser(
        description="Create GigaMIDI subsets - load all, filter, write (single pass)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
        choices=["s1", "s2", "s3", "s4"],
        help="Subset to create (s1, s2, s3, s4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory path",
    )
    parser.add_argument(
        "--sample_size",
        type=float,
        default=None,
        help="Sample percentage (for S1, S2, S3: e.g. 0.10 for 10%%)",
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
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for file writing (default: 8)",
    )

    args = parser.parse_args()

    # Validate sample_size for relevant subsets
    if args.subset in ["s1", "s2", "s3"] and args.sample_size is None:
        parser.error(f"--sample_size required for subset {args.subset}")

    sys.stdout.reconfigure(line_buffering=True)

    start_time = time.time()

    # Set sample size
    sample_size_val = args.sample_size if args.sample_size else 1.0
    if args.subset == "s4":
        sample_size_val = 1.0

    print("=" * 70)
    print("GigaMIDI Unified Subset Creator")
    print("=" * 70)
    subset_desc = get_subset_description(args.subset, sample_size_val)
    print(f"Subset:       {args.subset.upper()} ({subset_desc})")
    if sample_size_val < 1.0:
        print(f"Sample Size:  {sample_size_val * 100:.1f}%")
    else:
        print("Sample Size:  ALL")
    print(f"Seed:         {args.seed}")
    print(f"NOMML >=      {args.nomml_threshold}")
    print(f"Track Range:  {args.min_tracks}-{args.max_tracks}")
    print(f"Output:       {args.output}")
    print(f"Workers:      {args.workers}")
    if args.limit:
        print(f"Limit:        {args.limit:,} files (testing mode)")
    print("-" * 70)
    print("Loading entire dataset once (including MIDI binary)...")
    print("Single-pass filter + write to correct folder structure")
    print("-" * 70)
    sys.stdout.flush()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Stage 1: Load ENTIRE dataset (including MIDI binary)
    print("\n[Stage 1/5] Loading ENTIRE dataset (all splits, all MIDI binary)...")
    stage_start = time.time()

    all_records = []
    train_count = 0
    valid_count = 0
    test_count = 0

    for split_name in ["train", "validation", "test"]:
        print(f"\n  Loading {split_name} split...")

        if args.limit:
            ds = load_dataset(
                "Metacreation/GigaMIDI",
                "v2.0.0",
                split=split_name,
            )
            # Select limited range
            limit = min(args.limit, len(ds))
            ds = ds.select(range(limit))
        else:
            ds = load_dataset(
                "Metacreation/GigaMIDI",
                "v2.0.0",
                split=split_name,
            )

        # Convert to list
        records = list(ds)
        print(f"    Loaded {len(records):,} records")

        # Track split distribution
        for r in records:
            md5 = r.get("md5", "")
            if md5:
                first_char = md5[0].lower()
                if first_char in TRAIN_HASHES:
                    train_count += 1
                elif first_char in VALID_HASHES:
                    valid_count += 1
                elif first_char in TEST_HASHES:
                    test_count += 1

        all_records.extend(records)

    load_time = time.time() - stage_start
    print(f"\n  Total loaded: {len(all_records):,} records in {load_time:.1f}s")
    print(f"    - Train (0-d): {train_count:,}")
    print(f"    - Valid (e):   {valid_count:,}")
    print(f"    - Test (f):    {test_count:,}")
    sys.stdout.flush()

    # Stage 2: Filter in memory
    print("\n[Stage 2/5] Filtering in memory (single pass)...")
    stage_start = time.time()

    filtered = []
    with tqdm(
        total=len(all_records), desc="Filtering", unit="records", leave=True
    ) as pbar:
        for row in all_records:
            md5_val = row.get("md5", "")
            if not md5_val:
                pbar.update(1)
                continue

            # Apply subset-specific filtering
            if args.subset in ["s2", "s4"]:
                num_tracks = row.get("num_tracks", 0) or 0
                if num_tracks < args.min_tracks or num_tracks > args.max_tracks:
                    pbar.update(1)
                    continue

                nomml = row.get("NOMML", []) or []
                has_expressive = any(n >= args.nomml_threshold for n in nomml)

                if not has_expressive:
                    pbar.update(1)
                    continue

            # Get MIDI bytes
            midi_bytes = row.get("music", b"")
            if not midi_bytes:
                pbar.update(1)
                continue

            filtered.append(
                {
                    "md5": md5_val,
                    "music": midi_bytes,
                }
            )
            pbar.update(1)

    filter_time = time.time() - stage_start
    train_filtered = sum(1 for f in filtered if f["md5"][0].lower() in TRAIN_HASHES)
    valid_filtered = sum(1 for f in filtered if f["md5"][0].lower() in VALID_HASHES)
    test_filtered = sum(1 for f in filtered if f["md5"][0].lower() in TEST_HASHES)

    print(f"\n  Filtered in {filter_time:.1f}s")
    print(f"    - Total: {len(filtered):,}")
    print(f"    - Train: {train_filtered:,}")
    print(f"    - Valid: {valid_filtered:,}")
    print(f"    - Test:  {test_filtered:,}")

    # Free memory
    del all_records

    # Stage 3: Sort by md5
    print("\n[Stage 3/5] Sorting by MD5 for reproducibility...")
    stage_start = time.time()

    print(f"  Sorting {len(filtered):,} files by MD5...")
    with tqdm(total=len(filtered), desc="Sorting", unit="files", leave=True) as pbar:
        filtered.sort(key=lambda x: x["md5"])
        pbar.update(len(filtered))

    sort_time = time.time() - stage_start
    print(f"  Sorted in {sort_time:.1f}s")
    sys.stdout.flush()

    # Stage 4: Random sampling (if applicable)
    if args.subset in ["s1", "s2", "s3"]:
        print("\n[Stage 4/5] Random sampling...")
        stage_start = time.time()

        random.seed(args.seed)
        sample_size = int(len(filtered) * sample_size_val)

        print(f"  Total filtered: {len(filtered):,}")
        print(f"  Sample size: {sample_size_val * 100:.1f}% ({sample_size:,})")
        print(f"  Seed: {args.seed}")

        sampled = random.sample(filtered, k=sample_size)
        sample_time = time.time() - stage_start
        print(f"  Sampled in {sample_time:.1f}s")

        del filtered
    else:
        sampled = filtered

    print(f"\n  Subset: {len(sampled):,} files to write")

    train_final = sum(1 for f in sampled if f["md5"][0].lower() in TRAIN_HASHES)
    valid_final = sum(1 for f in sampled if f["md5"][0].lower() in VALID_HASHES)
    test_final = sum(1 for f in sampled if f["md5"][0].lower() in TEST_HASHES)

    print("\n  Split distribution:")
    train_pct = 100 * train_final / len(sampled)
    valid_pct = 100 * valid_final / len(sampled)
    test_pct = 100 * test_final / len(sampled)
    print(f"    - Train (0-d): {train_final:,} ({train_pct:.1f}%)")
    print(f"    - Valid (e):   {valid_final:,} ({valid_pct:.1f}%)")
    print(f"    - Test (f):    {test_final:,} ({test_pct:.1f}%)")

    # Stage 5: Write MIDI files in parallel
    print(f"\n[Stage 5/5] Writing MIDI files with {args.workers} workers...")
    stage_start = time.time()

    tasks = [(f["music"], f["md5"], output_path) for f in sampled]

    written = {"train": 0, "valid": 0, "test": 0}
    errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(write_midi_file, t): t[1] for t in tasks}

        with tqdm(total=len(futures), desc="Writing", unit="files", leave=True) as pbar:
            for future in as_completed(futures):
                try:
                    md5, split = future.result()
                    written[split] += 1
                except Exception:
                    errors += 1
                pbar.update(1)

    write_time = time.time() - stage_start
    print(f"\n  Written in {write_time:.1f}s ({write_time / 60:.1f} min)")
    print(f"    - Train: {written['train']:,}")
    print(f"    - Valid: {written['valid']:,}")
    print(f"    - Test:  {written['test']:,}")
    if errors > 0:
        print(f"    - Errors: {errors:,}")
    sys.stdout.flush()

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Subset:      {args.subset.upper()}")
    print(f"  Total files: {len(sampled):,}")
    print(f"    - Train:   {written['train']:,}")
    print(f"    - Valid:   {written['valid']:,}")
    print(f"    - Test:    {written['test']:,}")
    print(f"  Output dir:  {output_path}")
    print(f"  Total time:  {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)
    print("\nNote: File structure follows Lakh convention:")
    print("      output/train/{0-f}/ - md5 hash files")
    print("      output/valid/e/    - md5 hash files")
    print("      output/test/f/   - md5 hash files")
    print("\nNext steps:")
    print(
        "  1. Preprocess: python gigamidi_preprocess_to_compound.py --input <output>/"
    )
    print("  2. Tokenize:   python gigamidi_tokenize_events.py --input <output>/")
    print("  3. Define:    python gigamidi_define_splits.py --input <output>/")
    print("  4. Shuffle:   python gigamidi_shuffle_train.py --input <output>/")


if __name__ == "__main__":
    main()
