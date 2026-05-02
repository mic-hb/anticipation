#!/usr/bin/env python3
"""
GigaMIDI Unified Subset Creator (Direct from Cache)

This script creates GigaMIDI subsets (S1, S2, S3, S4) by loading from the HuggingFace
cache (already downloaded), filtering/sampling, and writing directly to the CORRECT
folder structure (train/valid/test by hash).

No separate download step needed - data is already cached!

Memory-efficient: processes in chunks to avoid OOM.

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
- Loads from HuggingFace cache (no download needed)
- Memory-efficient chunk processing
- Parallel MIDI file writing
- Verbose naming convention
- Hash-based train/valid/test split (Lakh convention)
- Supports S1, S2, S3, S4 subsets
- Deterministic sorting before sampling
- Full progress tracking with tqdm
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
    """Write a single MIDI file to the correct folder structure."""
    midi_bytes, md5, output_base = args_tuple

    split = get_split_from_hash(md5)
    hex_folder = get_hex_folder(md5)

    if split == "train":
        folder = output_base / "train" / hex_folder
    elif split == "valid":
        folder = output_base / "valid" / "e"
    else:
        folder = output_base / "test" / "f"

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
        description="Create GigaMIDI subsets directly from cache (no download needed)"
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
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10000,
        help="Number of rows per chunk for processing (default: 10000)",
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
    print("GigaMIDI Unified Subset Creator (Direct from Cache)")
    print("=" * 70)
    subset_desc = get_subset_description(args.subset, sample_size_val)
    print(f"Subset:       {args.subset.upper()} ({subset_desc})")
    if sample_size_val < 1.0:
        print(f"Sample Size:  {sample_size_val * 100:.1f}%")
    else:
        print("Sample Size:  ALL")
    print(f"Seed:        {args.seed}")
    print(f"NOMML >=     {args.nomml_threshold}")
    print(f"Track Range: {args.min_tracks}-{args.max_tracks}")
    print(f"Output:      {args.output}")
    print(f"Workers:     {args.workers}")
    print(f"Chunk Size:  {args.chunk_size:,}")
    if args.limit:
        print(f"Limit:       {args.limit:,} files (testing mode)")
    print("-" * 70)
    print("Note: Loading from HuggingFace cache (no download needed)")
    print("      Writing directly to train/valid/test structure")
    print("-" * 70)
    sys.stdout.flush()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # Stage 1: Collect md5s first (lightweight metadata only)
    print("\n[Stage 1/6] Collecting metadata (md5 list) with tqdm progress...")
    stage_start = time.time()

    all_md5s = []
    train_eligible = 0
    valid_eligible = 0
    test_eligible = 0

    for split_name in ["train", "validation", "test"]:
        print(f"\n  Processing {split_name} split...")

        ds = load_dataset(
            "Metacreation/GigaMIDI",
            "v2.0.0",
            split=split_name,
            streaming=True,
        )

        # Count total rows first (approximate)
        split_count = 0
        # We can't easily get total without loading, so track as we go
        # Use tqdm for this split
        with tqdm(desc=f"  {split_name}", unit="rows", leave=True) as pbar:
            for row in ds:
                if args.limit and split_count >= args.limit:
                    break

                md5_val = row.get("md5", "")
                if not md5_val:
                    split_count += 1
                    pbar.update(1)
                    continue

                # Quick filter check (without loading MIDI bytes)
                if args.subset in ["s2", "s4"]:
                    num_tracks = row.get("num_tracks", 0) or 0
                    if num_tracks < args.min_tracks or num_tracks > args.max_tracks:
                        split_count += 1
                        pbar.update(1)
                        continue

                    nomml = row.get("NOMML", []) or []
                    has_expressive = any(n >= args.nomml_threshold for n in nomml)

                    if not has_expressive:
                        split_count += 1
                        pbar.update(1)
                        continue

                all_md5s.append(
                    {
                        "md5": md5_val,
                        "split": split_name,
                    }
                )

                # Track split distribution
                first_char = md5_val[0].lower()
                if first_char in TRAIN_HASHES:
                    train_eligible += 1
                elif first_char in VALID_HASHES:
                    valid_eligible += 1
                elif first_char in TEST_HASHES:
                    test_eligible += 1

                split_count += 1
                pbar.update(1)

        print(f"    Found {split_count:,} rows processed")

    collect_time = time.time() - stage_start
    print(f"\n  Total eligible: {len(all_md5s):,} files in {collect_time:.1f}s")
    print(f"    - Train-eligible: {train_eligible:,}")
    print(f"    - Valid-eligible: {valid_eligible:,}")
    print(f"    - Test-eligible: {test_eligible:,}")
    sys.stdout.flush()

    # Stage 2: Sort by md5 for reproducibility
    print("\n[Stage 2/6] Sorting by MD5 for reproducibility...")
    stage_start = time.time()

    print(f"  Sorting {len(all_md5s):,} files by MD5...")
    with tqdm(total=len(all_md5s), desc="Sorting", unit="files", leave=True) as pbar:
        all_md5s.sort(key=lambda x: x["md5"])
        pbar.update(len(all_md5s))

    sort_time = time.time() - stage_start
    print(f"  Sorted in {sort_time:.1f}s")
    sys.stdout.flush()

    # Stage 3: Random sampling (if applicable)
    if args.subset in ["s1", "s2", "s3"]:
        print("\n[Stage 3/6] Random sampling...")
        stage_start = time.time()

        random.seed(args.seed)
        sample_size = int(len(all_md5s) * sample_size_val)

        print(f"  Total eligible: {len(all_md5s):,}")
        print(f"  Sample size: {sample_size_val * 100:.1f}% ({sample_size:,})")
        print(f"  Seed: {args.seed}")

        sampled_md5s = random.sample(all_md5s, k=sample_size)
        sample_time = time.time() - stage_start
        print(f"  Sampled in {sample_time:.1f}s")

        del all_md5s  # Free memory
    else:
        sampled_md5s = all_md5s

    print(f"\n  Subset: {len(sampled_md5s):,} files for writing")

    # Analyze splits
    train_count = sum(1 for f in sampled_md5s if f["md5"][0].lower() in TRAIN_HASHES)
    valid_count = sum(1 for f in sampled_md5s if f["md5"][0].lower() in VALID_HASHES)
    test_count = sum(1 for f in sampled_md5s if f["md5"][0].lower() in TEST_HASHES)

    train_pct = 100 * train_count / len(sampled_md5s)
    valid_pct = 100 * valid_count / len(sampled_md5s)
    test_pct = 100 * test_count / len(sampled_md5s)

    print("\n  Split distribution (will be written):")
    print(f"    - Train (0-d): {train_count:,} ({train_pct:.1f}%)")
    print(f"    - Valid (e):   {valid_count:,} ({valid_pct:.1f}%)")
    print(f"    - Test (f):    {test_count:,} ({test_pct:.1f}%)")

    # Stage 4: Create md5 -> index mapping for lookup
    print("\n[Stage 4/6] Building lookup index...")
    stage_start = time.time()

    md5_to_info = {f["md5"]: f for f in sampled_md5s}
    print(f"  Indexed {len(md5_to_info):,} files in {time.time() - stage_start:.1f}s")
    sys.stdout.flush()

    # Stage 5: Load and write MIDI files in chunks
    print(
        f"\n[Stage 5/6] Loading and writing MIDI files with {args.workers} workers..."
    )
    stage_start = time.time()

    written = {"train": 0, "valid": 0, "test": 0}
    errors = 0

    # Process each split separately with progress
    for split_name in ["train", "validation", "test"]:
        print(f"\n  Processing {split_name} split...")

        ds = load_dataset(
            "Metacreation/GigaMIDI",
            "v2.0.0",
            split=split_name,
            streaming=True,
        )

        split_tasks = []
        split_written = 0

        # Process with tqdm
        with tqdm(desc=f"  {split_name}", unit="rows", leave=True) as pbar:
            for row in ds:
                md5_val = row.get("md5", "")
                if md5_val not in md5_to_info:
                    pbar.update(1)
                    continue

                midi_bytes = row.get("music", b"")
                if not midi_bytes:
                    pbar.update(1)
                    continue

                split_tasks.append((midi_bytes, md5_val, output_path))

                # Process in chunks
                if len(split_tasks) >= args.chunk_size:
                    with ThreadPoolExecutor(max_workers=args.workers) as executor:
                        futures = [
                            executor.submit(write_midi_file, t) for t in split_tasks
                        ]

                        for future in as_completed(futures):
                            try:
                                md5, split = future.result()
                                written[split] += 1
                            except Exception:
                                errors += 1

                    split_written += len(split_tasks)
                    pbar.update(len(split_tasks))
                    split_tasks = []

        # Process remaining
        if split_tasks:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(write_midi_file, t) for t in split_tasks]

                for future in as_completed(futures):
                    try:
                        md5, split = future.result()
                        written[split] += 1
                    except Exception:
                        errors += 1

            split_written += len(split_tasks)

    write_time = time.time() - stage_start
    print(f"\n  Written in {write_time:.1f}s ({write_time / 60:.1f} min)")
    print(f"    - Train: {written['train']:,}")
    print(f"    - Valid: {written['valid']:,}")
    print(f"    - Test: {written['test']:,}")
    if errors > 0:
        print(f"    - Errors: {errors:,}")
    sys.stdout.flush()

    # Stage 6: Verify output
    print("\n[Stage 6/6] Verifying output...")

    total_files = 0
    for split in ["train", "valid", "test"]:
        split_dir = output_path / split
        if split_dir.exists():
            for hex_dir in split_dir.iterdir():
                if hex_dir.is_dir():
                    files = list(hex_dir.glob("*.mid"))
                    total_files += len(files)

    print(f"  Total MIDI files written: {total_files:,}")

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Subset:          {args.subset.upper()}")
    print(f"  Total files:    {total_files:,}")
    print(f"    - Train:     {written['train']:,}")
    print(f"    - Valid:     {written['valid']:,}")
    print(f"    - Test:      {written['test']:,}")
    print(f"  Output dir:    {output_path}")
    print(f"  Total time:   {total_time:.1f}s ({total_time / 60:.1f} min)")
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
