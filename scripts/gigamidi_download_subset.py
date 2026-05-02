#!/usr/bin/env python3
"""
GigaMIDI Subset Downloader (All Splits)

Downloads MIDI files from GigaMIDI based on md5 list from subset creation.
Handles files from ALL splits (train, validation, test).

The JSON should contain a "split" field indicating which GigaMIDI split each file came from.

Usage:
    python scripts/gigamidi_download_subset.py \
        --input data/gigamidi_s1_10pct_random_from_all.json \
        --output data/gigamidi_s1_10pct_random_from_all/

Features:
- Progress bar with tqdm
- Time estimates
- Download speed tracking
- File size tracking
- Resumable (skips existing files)
- Downloads from appropriate split based on JSON metadata
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def download_single(args_tuple):
    """Download a single file. Called by thread pool."""
    md5, output_dir, skip_existing = args_tuple

    existing_file = output_dir / f"{md5}.mid"
    if skip_existing and existing_file.exists():
        return md5, existing_file.stat().st_size, True

    return md5, 0, False


def main():
    parser = argparse.ArgumentParser(
        description="Download MIDI files from GigaMIDI using md5 list (all splits)"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON file with md5 list (from subset creation scripts)",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory to save MIDI files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to download (for testing)",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip files that already exist (default: True)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers (default: 8)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    start_time = time.time()

    print("=" * 70)
    print("GigaMIDI Subset Downloader (All Splits) - Parallel")
    print("=" * 70)
    print(f"Input:    {args.input}")
    print(f"Output:   {args.output}")
    print(f"Workers: {args.workers}")
    if args.limit:
        print(f"Limit:   {args.limit:,} files (testing mode)")
    print(f"Skip existing: {args.skip_existing}")
    print("-" * 70)
    sys.stdout.flush()

    # Stage 1: Load md5 list and create download tasks
    print("\n[Stage 1/4] Loading MD5 list and building download queue...")
    stage_start = time.time()

    with open(args.input) as f:
        file_list = json.load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    download_queue = []
    for item in file_list:
        md5 = item.get("md5", "")
        if not md5 or (args.limit and len(download_queue) >= args.limit):
            continue
        download_queue.append((md5, output_dir, args.skip_existing))

    queue_size = len(download_queue)
    stage_time = time.time() - stage_start
    print(f"  Built queue with {queue_size:,} files in {stage_time:.1f}s")
    sys.stdout.flush()

    # Check existing files to skip
    if args.skip_existing:
        print("\n[Stage 2/4] Checking existing files...")
        stage_start = time.time()

        existing_md5s = {f.stem for f in output_dir.glob("*.mid")}
        new_queue = [t for t in download_queue if t[0] not in existing_md5s]
        skipped = len(download_queue) - len(new_queue)

        stage_time = time.time() - stage_start
        print(f"  Skipping {skipped:,} existing files")
        print(f"  Remaining: {len(new_queue):,}")
        download_queue = new_queue

    # Stage 3: Download with thread pool
    print(f"\n[Stage 3/4] Downloading with {args.workers} workers...")
    stage_start = time.time()

    downloaded = 0
    failed = 0
    total_size = 0

    def download_with_dataset(args_tuple):
        md5, output_dir, skip = args_tuple
        try:
            for split_name in ["train", "validation", "test"]:
                ds = load_dataset(
                    "Metacreation/GigaMIDI", "v2.0.0", split=split_name, streaming=True
                )
                for row in ds:
                    if row.get("md5", "") != md5:
                        continue
                    music_data = row.get("music")
                    if not music_data:
                        return md5, 0, False
                    midi_bytes = music_data.get("bytes")
                    if not midi_bytes:
                        return md5, 0, False
                    existing_file = output_dir / f"{md5}.mid"
                    with open(existing_file, "wb") as f:
                        f.write(midi_bytes)
                    return md5, len(midi_bytes), True
        except Exception:
            pass
        return md5, 0, False

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_with_dataset, t): t for t in download_queue}

        with tqdm(total=len(futures), desc="Downloading", unit="files") as pbar:
            for future in as_completed(futures):
                md5, size, success = future.result()
                if success:
                    downloaded += 1
                    total_size += size
                else:
                    failed += 1
                pbar.update(1)

                if downloaded % 1000 == 0:
                    elapsed = time.time() - stage_start
                    rate = downloaded / elapsed if elapsed > 0 else 0
                    remaining = len(futures) - downloaded - failed
                    eta = remaining / rate if rate > 0 else 0
                    pbar.set_postfix(
                        {"rate": f"{rate:.0f}f/s", "ETA": f"{eta / 60:.1f}m"}
                    )

    download_time = time.time() - stage_start
    rate = downloaded / download_time if download_time > 0 else 0
    print(
        f"\n  Downloaded: {downloaded:,} files in {download_time:.1f}s ({rate:.0f} files/s)"
    )
    print(f"  Failed:   {failed:,}")
    print(f"  Total size: {total_size / (1024**3):.2f} GB")
    sys.stdout.flush()

    # Stage 4: Verify
    print("\n[Stage 4/4] Verifying downloads...")
    stage_start = time.time()

    actual_files = list(output_dir.glob("*.mid"))
    verified = len(actual_files)

    verify_time = time.time() - stage_start
    print(f"  Verified: {verified:,} files")
    sys.stdout.flush()

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Files downloaded:    {downloaded:,}")
    print(f"  Files verified:  {verified:,}")
    print(f"  Total size:   {total_size / (1024**3):.2f} GB")
    print(f"  Output:     {output_dir}")
    print(f"  Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count existing files
    existing = len(list(output_dir.glob("*.mid")))
    if args.skip_existing and existing > 0:
        print(f"  Found {existing:,} existing files (skipping)")
        for split in splits:
            splits[split] = splits[split] - {f.stem for f in output_dir.glob("*.mid")}
        total_target = sum(len(v) for v in splits.values())
        print(f"  Remaining to download: {total_target:,}")

    print("-" * 70)

    # Download from each split
    print("\n[Stage 2/4] Downloading MIDI files...")
    stage_start = time.time()
    sys.stdout.flush()

    downloaded = 0
    skipped = 0
    total_size = 0

    for split_name, target_md5s in splits.items():
        if not target_md5s:
            continue

        print(
            f"\n  Downloading from {split_name} split ({len(target_md5s):,} files)..."
        )

        ds = load_dataset(
            "Metacreation/GigaMIDI", "v2.0.0", split=split_name, streaming=True
        )

        for row in ds:
            md5 = row.get("md5", "")

            if md5 not in target_md5s:
                continue

            existing_file = output_dir / f"{md5}.mid"
            if args.skip_existing and existing_file.exists():
                skipped += 1
                total_size += existing_file.stat().st_size
                downloaded += 1
                continue

            music_data = row.get("music")
            if not music_data:
                skipped += 1
                continue

            midi_bytes = music_data.get("bytes")
            if not midi_bytes:
                skipped += 1
                continue

            with open(existing_file, "wb") as f:
                f.write(midi_bytes)

            file_size = len(midi_bytes)
            total_size += file_size
            downloaded += 1

            if downloaded % 1000 == 0:
                elapsed = time.time() - stage_start
                rate = downloaded / elapsed if elapsed > 0 else 0
                remaining = total_target - downloaded
                eta = remaining / rate if rate > 0 else 0
                print(
                    f"    {downloaded:,}/{total_target:,} | {rate:.0f}f/s | ETA: {eta / 60:.1f}m"
                )

            if args.limit and downloaded >= args.limit:
                break

            if downloaded >= total_target + skipped:
                break

        if args.limit and downloaded >= args.limit:
            break

    download_time = time.time() - stage_start
    print(f"\n  Downloaded: {downloaded:,} files in {download_time:.1f}s")
    print(f"  Skipped:   {skipped:,} files")
    print(f"  Speed:    {downloaded / download_time:.0f} files/s")
    sys.stdout.flush()

    # Verify
    print("\n[Stage 3/4] Verifying downloads...")
    stage_start = time.time()

    actual_files = list(output_dir.glob("*.mid"))
    verified = len(actual_files)

    total_verify_size = sum(f.stat().st_size for f in actual_files)

    verify_time = time.time() - stage_start
    print(f"  Verified: {verified:,} files")
    print(f"  Size:   {total_verify_size / (1024**3):.2f} GB")
    sys.stdout.flush()

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Files downloaded:    {downloaded:,}")
    print(f"  Files skipped:        {skipped:,}")
    print(f"  Files verified:      {verified:,}")
    print(f"  Total size:       {total_verify_size / (1024**3):.2f} GB")
    print(f"  Output:          {output_dir}")
    print(f"  Total time:       {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)
    print("\nNote: After download, use restructure script for hash-based splitting:")
    print("       0-d -> train/, e -> valid/, f -> test/")


if __name__ == "__main__":
    main()
