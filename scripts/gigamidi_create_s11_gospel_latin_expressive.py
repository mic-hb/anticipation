#!/usr/bin/env python3
"""
GigaMIDI Subset S11: Gospel + Latin Expressive

Filters GigaMIDI to files where:
  1. music_styles_curated contains "gospel" OR "latin" (case-insensitive), AND
  2. At least one track has NOMML >= 12 (expressive performance)

Then writes to hex-folder structure (Lakh convention):
    output/0-f/  (train: md5 starts with 0-9 or a-d)
    output/e/     (valid: md5 starts with e)
    output/f/     (test: md5 starts with f)

S11 Definition (from docs/amt-fine-tuning.md):
- Subset ID: S11
- Description: Gospel + Latin Expressive
- Files: Varies (genre + NOMML-filtered subset)
- Selection Method: Style = gospel OR latin AND NOMML >= 12

Usage:
    python scripts/gigamidi_create_s11_gospel_latin_expressive.py \
        --output data/gigamidi_s11_gospel_latin_expressive/ \
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


def stream_and_write(output_path, workers, nomml_threshold=12, limit=None):
    """Stream all splits, filter for gospel/latin + NOMML>=threshold, write immediately."""
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

            # Genre filter: gospel OR latin
            styles = row.get("music_styles_curated", []) or []
            if not any(s.lower() in ("gospel", "latin") for s in styles):
                continue

            # NOMML expressive filter: at least one track with NOMML >= threshold
            nomml = row.get("NOMML", []) or []
            if not any(n >= nomml_threshold for n in nomml):
                continue

            midi_bytes = row.get("music", b"")
            if not midi_bytes:
                continue

            tasks.append((midi_bytes, md5_val, output_path))

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
        description="Create S11: Gospel + Latin Expressive subset from GigaMIDI"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gigamidi_s11_gospel_latin_expressive/",
        help="Output directory path",
    )
    parser.add_argument(
        "--nomml_threshold",
        type=int,
        default=12,
        help="NOMML threshold for expressive tracks (default: 12)",
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
    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)
    start_time = time.time()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("GigaMIDI Subset S11 — Gospel + Latin Expressive")
    print("=" * 70)
    print(f"Output:         {output_path}")
    print(f"NOMML >=        {args.nomml_threshold}")
    print(f"Workers:        {args.workers}")
    if args.limit:
        print(f"Limit:          {args.limit:,} per split (TEST MODE)")
    print("-" * 70)
    print("Streaming: one split at a time, immediate write, minimal RAM")
    print("Filter: style = gospel OR latin AND NOMML >= {args.nomml_threshold}")
    print("=" * 70)

    written, errors = stream_and_write(
        output_path, args.workers, args.nomml_threshold, args.limit
    )
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
