#!/usr/bin/env python3
"""
GigaMIDI Train Subset Downloader

Downloads MIDI files from GigaMIDI train split based on md5 list from subset creation.

Usage:
    python scripts/gigamidi_download_train_subset.py \
        --input data/gigamidi_s1_10pct_random_from_all.json \
        --output data/gigamidi_s1_10pct_random_from_all/ \
        --limit 1000

Features:
- Progress bar with tqdm
- Time estimates
- Download speed tracking
- File size tracking
- Resumable (skips existing files)
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
        description="Download MIDI files from GigaMIDI train split using md5 list"
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
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Which GigaMIDI split to download from (default: train)",
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

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    start_time = time.time()

    print("=" * 70)
    print("GigaMIDI Train Subset Downloader")
    print("=" * 70)
    print(f"Input:    {args.input}")
    print(f"Output:   {args.output}")
    print(f"Split:   {args.split}")
    if args.limit:
        print(f"Limit:   {args.limit:,} files (testing mode)")
    print(f"Skip existing: {args.skip_existing}")
    print("-" * 70)
    sys.stdout.flush()

    # Load md5 list
    print("\n[Stage 1/3] Loading MD5 list...")
    stage_start = time.time()

    with open(args.input) as f:
        file_list = json.load(f)

    target_md5s = {item["md5"] for item in file_list}
    total_target = len(target_md5s)

    load_time = time.time() - stage_start
    print(f"  Loaded {total_target:,} MD5s in {load_time:.1f}s")
    sys.stdout.flush()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count existing files
    existing = len(list(output_dir.glob("*.mid")))
    if args.skip_existing and existing > 0:
        print(f"  Found {existing:,} existing files (skipping)")
        target_md5s = target_md5s - {f.stem for f in output_dir.glob("*.mid")}
        total_target = len(target_md5s)
        print(f"  Remaining to download: {total_target:,}")

    print("-" * 70)

    # Download from GigaMIDI
    print("\n[Stage 2/3] Downloading MIDI files...")
    stage_start = time.time()
    sys.stdout.flush()

    ds = load_dataset(
        "Metacreation/GigaMIDI", "v2.0.0", split=args.split, streaming=True
    )

    downloaded = 0
    skipped = 0
    total_size = 0

    with tqdm(
        total=total_target,
        desc="Downloading",
        unit="files",
        unit_scale=True,
        unit_divisor=1000,
    ) as pbar:
        for row in ds:
            md5 = row.get("md5", "")

            if md5 not in target_md5s:
                continue

            existing_file = output_dir / f"{md5}.mid"
            if args.skip_existing and existing_file.exists():
                skipped += 1
                total_size += existing_file.stat().st_size
                downloaded += 1
                pbar.update(1)
                if args.limit and downloaded >= args.limit:
                    break
                continue

            music_data = row.get("music")
            if not music_data:
                skipped += 1
                pbar.update(1)
                continue

            midi_bytes = music_data.get("bytes")
            if not midi_bytes:
                skipped += 1
                pbar.update(1)
                continue

            with open(existing_file, "wb") as f:
                f.write(midi_bytes)

            file_size = len(midi_bytes)
            total_size += file_size
            downloaded += 1

            pbar.update(1)

            if downloaded % 1000 == 0 or downloaded == total_target:
                elapsed = time.time() - stage_start
                rate = downloaded / elapsed if elapsed > 0 else 0
                remaining = total_target - downloaded
                eta = remaining / rate if rate > 0 else 0
                pbar.set_postfix(
                    {
                        "rate": f"{rate:.0f}f/s",
                        "size": f"{total_size / (1024**3):.2f}GB",
                        "ETA": f"{eta / 60:.1f}m" if eta > 0 else "done",
                    }
                )

            if args.limit and downloaded >= args.limit:
                break

            if downloaded >= total_target + skipped:
                break

    download_time = time.time() - stage_start
    print(f"\n  Downloaded: {downloaded:,} files in {download_time:.1f}s")
    print(f"  Skipped:   {skipped:,} files")
    print(f"  Speed:    {(downloaded) / download_time:.0f} files/s")
    sys.stdout.flush()

    # Verify
    print("\n[Stage 3/3] Verifying downloads...")
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


if __name__ == "__main__":
    main()
