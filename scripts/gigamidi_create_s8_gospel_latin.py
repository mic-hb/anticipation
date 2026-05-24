#!/usr/bin/env python3
"""
GigaMIDI Subset S8: Gospel + Latin Genre

Filters GigaMIDI to files where music_styles_curated contains
"gospel" or "latin" (case-insensitive).

Then writes to hex-folder structure (Lakh convention):
    output/0-f/  (train: md5 starts with 0-9 or a-d)
    output/e/     (valid: md5 starts with e)
    output/f/     (test: md5 starts with f)

S8 Definition (from docs/amt-fine-tuning.md):
- Subset ID: S8
- Description: Gospel + Latin genre
- Files: Varies (genre-filtered subset)
- Selection Method: Style = gospel OR latin (Musicmap topology)

Usage:
    python scripts/gigamidi_create_s8_gospel_latin.py \
        --output data/gigamidi_s8_gospel_latin/ \
        --workers 16

Features:
- Streaming dataset load (generator, NOT list()) — RAM-safe for Colab
- Parallel MIDI file writing
- Full progress tracking with tqdm
- Hash-based train/valid/test split (Lakh convention)
"""

import argparse
import gc
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


TRAIN_HASHES = set("0123456789abcd")
VALID_HASHES = set("e")
TEST_HASHES = set("f")


def get_hex_folder(md5: str) -> str:
    return md5[0].lower()


def write_midi_file(args_tuple):
    """Write a single MIDI file to hex folder."""
    midi_bytes, md5_val, output_base = args_tuple
    hex_folder = get_hex_folder(md5_val)
    folder = output_base / hex_folder
    folder.mkdir(parents=True, exist_ok=True)
    midi_path = folder / f"{md5_val}.mid"
    with open(midi_path, "wb") as f:
        f.write(midi_bytes)
    split = (
        "train" if md5_val[0].lower() in TRAIN_HASHES else
        "valid" if md5_val[0].lower() in VALID_HASHES else "test"
    )
    return md5_val, split


def stream_and_write(output_path, workers, dry_run=False, limit=None):
    """Stream all splits, filter for gospel/latin, write immediately."""
    written = {"train": 0, "valid": 0, "test": 0}
    errors = 0

    for split_name in ["train", "validation", "test"]:
        print(f"\n  [{split_name}] Loading in streaming mode...")
        ds = load_dataset(
            "Metacreation/GigaMIDI",
            "v2.0.0",
            split=split_name,
            streaming=True,
        )

        tasks = []
        scanned = 0
        for row in tqdm(ds, desc=f"  Filter {split_name}", total=limit, leave=True):
            if limit and scanned >= limit:
                break
            scanned += 1

            md5_val = row.get("md5", "")
            if not md5_val:
                continue

            styles = row.get("music_styles_curated", []) or []
            if not any(s.lower() in ("gospel", "latin") for s in styles):
                continue

            midi_bytes = row.get("music", b"")
            if not midi_bytes:
                continue

            tasks.append((midi_bytes, md5_val, output_path))

        if dry_run:
            print(f"  [{split_name}] DRY RUN — would write {len(tasks):,} files")
            del ds
            continue

        print(f"  [{split_name}] Writing {len(tasks):,} files...")
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(write_midi_file, t): t[1] for t in tasks}
            for future in tqdm(futures, desc=f"  Write {split_name}", leave=True):
                try:
                    md5, split = future.result()
                    written[split] += 1
                except Exception:
                    errors += 1

        del tasks, ds
        gc.collect()

    return written, errors


def main():
    parser = argparse.ArgumentParser(
        description="Create S8: Gospel + Latin genre subset from GigaMIDI"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gigamidi_s8_gospel_latin/",
        help="Output directory path",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=16,
        help="Parallel workers for file writing (default: 16)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit records per split (TEST MODE)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Scan and print statistics without downloading or writing any files",
    )
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    start_time = time.time()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GigaMIDI Subset S8 — Gospel + Latin Genre")
    print("=" * 70)
    print(f"Output:  {output_path}")
    print(f"Workers: {args.workers}")
    if args.limit:
        print(f"Limit:   {args.limit:,} per split (TEST MODE)")
    print("-" * 70)
    print("Streaming: one split at a time, immediate write, minimal RAM")
    print("Filter: music_styles_curated contains 'gospel' OR 'latin'")
    if args.dry_run:
        print("*** DRY RUN MODE — no files will be written ***")
    print("=" * 70)

    written, errors = stream_and_write(output_path, args.workers, dry_run=args.dry_run, limit=args.limit)
    elapsed = time.time() - start_time

    total_written = sum(written.values())

    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Files written: {total_written:,}")
    print(f"  Write errors:  {errors:,}")
    print(f"    Train (0-d): {written['train']:,}")
    print(f"    Valid (e):   {written['valid']:,}")
    print(f"    Test (f):   {written['test']:,}")
    print(f"  Output:        {output_path}")
    print(f"  Time:          {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Preprocess: python gigamidi_preprocess_to_compound.py --input <output>/")
    print("  2. Tokenize:   python gigamidi_tokenize_events.py --input <output>/")
    print("  3. Define:    python gigamidi_define_splits.py --input <output>/")
    print("  4. Shuffle:   python gigamidi_shuffle_train.py --input <output>/")


if __name__ == "__main__":
    main()
