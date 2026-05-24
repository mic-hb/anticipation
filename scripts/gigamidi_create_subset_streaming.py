#!/usr/bin/env python3
"""
GigaMIDI Subset Creator — STREAMING / LOW-MEMORY VERSION

Loads GigaMIDI with streaming=True (generator, not list),
filters records one at a time, writes files immediately.
NO list() materialization. NO in-memory sort.

For Colab instances with limited RAM (~12GB) where loading the
entire dataset into memory would crash the kernel.

Supported subsets: s1, s2, s3, s4, s8, s11

Usage (S8 / S11 — genre subsets, no sampling):
    python gigamidi_create_subset_streaming.py \
        --subset s8 \
        --output data/gigamidi_s8_gospel_latin/ \
        --workers 16

    python gigamidi_create_subset_streaming.py \
        --subset s11 \
        --nomml_threshold 12 \
        --output data/gigamidi_s11_gospel_latin_expressive/ \
        --workers 16

Usage (S1 / S3 — random sample):
    python gigamidi_create_subset_streaming.py \
        --subset s1 \
        --sample_size 0.10 \
        --seed 42 \
        --output data/gigamidi_s1_10pct_random_from_all/ \
        --workers 16

Usage (S2 / S4 — expressive subsets):
    python gigamidi_create_subset_streaming.py \
        --subset s4 \
        --nomml_threshold 12 \
        --output data/gigamidi_s4_all_expressive/ \
        --workers 16

For sampling subsets (S1/S2/S3): files are written to disk first, then
a random sample of the written files is retained. This avoids holding
all accepted file metadata in memory.

The tradeoff: writing all files then deleting the oversample means
temporary disk usage = subset_size × file_size. For 10% sampling of
1.4M files this is manageable (~100-200GB temp space on Colab).
"""

import argparse
import gc
import random
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
    """Write a single MIDI file to hex folder (flat Lakh-style structure)."""
    midi_bytes, md5, output_base = args_tuple
    hex_folder = get_hex_folder(md5)
    folder = output_base / hex_folder
    folder.mkdir(parents=True, exist_ok=True)
    midi_path = folder / f"{md5}.mid"
    with open(midi_path, "wb") as f:
        f.write(midi_bytes)
    split = (
        "train" if md5[0].lower() in TRAIN_HASHES else
        "valid" if md5[0].lower() in VALID_HASHES else
        "test"
    )
    return md5, split


def filter_record(row, subset, nomml_threshold, min_tracks, max_tracks):
    """Return True if the record passes the subset filter, False otherwise."""
    md5_val = row.get("md5", "")
    if not md5_val:
        return False, None, None

    if subset == "s8":
        styles = row.get("music_styles_curated", []) or []
        if not any(s.lower() in ("gospel", "latin") for s in styles):
            return False, None, None

    elif subset == "s11":
        styles = row.get("music_styles_curated", []) or []
        if not any(s.lower() in ("gospel", "latin") for s in styles):
            return False, None, None
        nomml = row.get("NOMML", []) or []
        if not any(n >= nomml_threshold for n in nomml):
            return False, None, None

    elif subset in ("s2", "s4"):
        num_tracks = row.get("num_tracks", 0) or 0
        if num_tracks < min_tracks or num_tracks > max_tracks:
            return False, None, None
        nomml = row.get("NOMML", []) or []
        if not any(n >= nomml_threshold for n in nomml):
            return False, None, None

    # s1/s3 accept all (sampling handled post-write)
    # s4 accepts all after track/nomml check above

    midi_bytes = row.get("music", b"")
    if not midi_bytes:
        return False, None, None

    return True, md5_val, midi_bytes


def stream_split_write(split_name, subset, output_base, nomml_threshold,
                      min_tracks, max_tracks, workers, sample_size=None,
                      seed=42, limit=None):
    """
    Stream a single GigaMIDI split, filter, write immediately.
    For sampled subsets (s1/s3): writes all, then deletes oversample.
    Returns (written, errors, sampled_and_kept) counts.
    """
    print(f"\n  [{split_name}] Opening streaming dataset...")
    ds = load_dataset(
        "Metacreation/GigaMIDI",
        "v2.0.0",
        split=split_name,
        streaming=True,
    )

    # For sampled subsets: collect all md5s first pass, then second pass write
    if sample_size and sample_size < 1.0:
        # Two-pass: collect md5s → sample → write
        print(f"  [{split_name}] Two-pass sampling ({sample_size * 100:.0f}%)...")
        all_accepted = []

        # Pass 1: collect accepted md5s
        scanned = 0
        for row in tqdm(ds, desc=f"  Scan {split_name}", total=limit, leave=True):
            if limit and scanned >= limit:
                break
            scanned += 1

            passed, md5_val, _ = filter_record(
                row, subset, nomml_threshold, min_tracks, max_tracks
            )
            if passed:
                all_accepted.append(md5_val)

        print(f"  [{split_name}] Accepted {len(all_accepted):,} files (scanned {scanned:,})")

        if not all_accepted:
            return 0, 0, 0

        # Random sample
        random.seed(seed)
        sample_count = max(1, int(len(all_accepted) * sample_size))
        sampled_md5s = set(random.sample(all_accepted, k=sample_count))
        del all_accepted  # free memory

        print(f"  [{split_name}] Sampled {sample_count:,} files. Writing...")
        del ds
        gc.collect()

        # Pass 2: re-stream and write only sampled
        ds = load_dataset(
            "Metacreation/GigaMIDI",
            "v2.0.0",
            split=split_name,
            streaming=True,
        )
        written = 0
        errors = 0
        for row in tqdm(ds, desc=f"  Write {split_name}", total=sample_count, leave=True):
            if row.get("md5", "") not in sampled_md5s:
                continue
            md5_val = row["md5"]
            midi_bytes = row.get("music", b"")
            if not midi_bytes:
                continue
            tasks = [(midi_bytes, md5_val, output_base)]
            with ThreadPoolExecutor(max_workers=1) as ex:
                f = ex.submit(write_midi_file, tasks[0])
                try:
                    f.result()
                    written += 1
                except Exception:
                    errors += 1
        del ds
        gc.collect()
        return written, errors, sample_count

    # Single-pass for non-sampled subsets (s4, s8, s11)
    print(f"  [{split_name}] Single-pass filter+write...")
    written = 0
    errors = 0
    scanned = 0

    for row in tqdm(ds, desc=f"  {split_name}", total=limit, leave=True):
        if limit and scanned >= limit:
            break
        scanned += 1

        passed, md5_val, midi_bytes = filter_record(
            row, subset, nomml_threshold, min_tracks, max_tracks
        )
        if not passed:
            continue

        # Write immediately
        try:
            folder = output_base / get_hex_folder(md5_val)
            folder.mkdir(parents=True, exist_ok=True)
            midi_path = folder / f"{md5_val}.mid"
            with open(midi_path, "wb") as f:
                f.write(midi_bytes)
            written += 1
        except Exception:
            errors += 1

    del ds
    gc.collect()
    return written, errors, 0  # 0 = no sampling


def main():
    parser = argparse.ArgumentParser(
        description="Create GigaMIDI subsets — STREAMING (low memory, Colab-safe)"
    )
    parser.add_argument(
        "--subset", type=str, required=True,
        choices=["s1", "s2", "s3", "s4", "s8", "s11"],
        help="Subset to create",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory path (hex-folder structure, Lakh convention)",
    )
    parser.add_argument(
        "--sample_size", type=float, default=None,
        help="For S1/S2/S3: fraction to sample (e.g. 0.10 for 10%%)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--nomml_threshold", type=int, default=12,
        help="NOMML threshold for expressive subsets (default: 12)",
    )
    parser.add_argument(
        "--min_tracks", type=int, default=1,
        help="Minimum number of tracks (default: 1)",
    )
    parser.add_argument(
        "--max_tracks", type=int, default=16,
        help="Maximum number of tracks (default: 16)",
    )
    parser.add_argument(
        "--workers", type=int, default=8,
        help="Parallel workers for file writing (default: 8)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit records per split (TEST MODE — for quick validation)",
    )
    args = parser.parse_args()

    # Validate
    if args.subset in ["s1", "s2", "s3"] and args.sample_size is None:
        parser.error(f"--sample_size required for subset {args.subset}")

    sys.stdout.reconfigure(line_buffering=True)
    start_time = time.time()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    _pct = f"{int(args.sample_size * 100)}%" if args.sample_size else "ALL"
    subset_descs = {
        "s1": "10% random from all",
        "s2": f"{_pct} random from expressive",
        "s3": f"{_pct} random from all",
        "s4": "ALL expressive files",
        "s8": "Gospel + Latin genre",
        "s11": "Gospel + Latin Expressive (genre + NOMML >= 12)",
    }

    print("=" * 70)
    print("GigaMIDI Subset Creator — STREAMING (low memory)")
    print("=" * 70)
    print(f"Subset:       {args.subset.upper()} ({subset_descs[args.subset]})")
    print(f"Output:       {output_path}")
    print(f"Workers:      {args.workers}")
    if args.sample_size:
        print(f"Sample:       {args.sample_size * 100:.0f}% (seed={args.seed})")
    print(f"NOMML >=      {args.nomml_threshold}")
    print(f"Track range:  {args.min_tracks}-{args.max_tracks}")
    if args.limit:
        print(f"Limit:        {args.limit:,} per split (TEST MODE)")
    print("-" * 70)
    print("Streaming: one split at a time, immediate write, minimal RAM")
    print("=" * 70)

    total_written = 0
    total_errors = 0

    for split_name in ["train", "validation", "test"]:
        w, e, _ = stream_split_write(
            split_name=split_name,
            subset=args.subset,
            output_base=output_path,
            nomml_threshold=args.nomml_threshold,
            min_tracks=args.min_tracks,
            max_tracks=args.max_tracks,
            workers=args.workers,
            sample_size=args.sample_size,
            seed=args.seed,
            limit=args.limit,
        )
        total_written += w
        total_errors += e

    elapsed = time.time() - start_time

    # Final counts
    train_count = (
        len(list(output_path.glob("0/*"))) +
        len(list(output_path.glob("1/*"))) +
        len(list(output_path.glob("2/*"))) +
        len(list(output_path.glob("3/*"))) +
        len(list(output_path.glob("4/*"))) +
        len(list(output_path.glob("5/*"))) +
        len(list(output_path.glob("6/*"))) +
        len(list(output_path.glob("7/*"))) +
        len(list(output_path.glob("8/*"))) +
        len(list(output_path.glob("9/*"))) +
        len(list(output_path.glob("a/*"))) +
        len(list(output_path.glob("b/*"))) +
        len(list(output_path.glob("c/*"))) +
        len(list(output_path.glob("d/*")))
    )
    valid_count = len(list(output_path.glob("e/*")))
    test_count = len(list(output_path.glob("f/*")))

    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Files written:  {total_written:,}")
    print(f"  Write errors:  {total_errors:,}")
    print(f"    Train (0-d): {train_count:,}")
    print(f"    Valid (e):   {valid_count:,}")
    print(f"    Test (f):   {test_count:,}")
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
