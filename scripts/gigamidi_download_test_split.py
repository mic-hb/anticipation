#!/usr/bin/env python3
"""
GigaMIDI Test Split Downloader

Downloads all MIDI files from the GigaMIDI test split.

Usage:
    python scripts/gigamidi_download_test_split.py \
        --output data/gigamidi_test_raw/ \
        --limit 1000

Features:
- Progress bar with tqdm
- Time estimates
- Download speed tracking
- File size tracking
- Skips existing files
"""

import argparse
import sys
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Download GigaMIDI test split")
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
        help="Limit number of files (for testing)",
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
    print("GigaMIDI Test Split Downloader")
    print("=" * 70)
    print(f"Output:  {args.output}")
    if args.limit:
        print(f"Limit:  {args.limit:,} files (testing mode)")
    print(f"Skip existing: {args.skip_existing}")
    print("-" * 70)
    sys.stdout.flush()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: Count total files
    print("\n[Stage 1/3] Counting test files...")
    stage_start = time.time()

    ds = load_dataset("Metacreation/GigaMIDI", "v2.0.0", split="test", streaming=True)

    total = 0
    for _ in ds:
        total += 1

    count_time = time.time() - stage_start
    print(f"  Total test files: {total:,} in {count_time:.1f}s")
    sys.stdout.flush()

    # Check existing
    existing = 0
    if args.skip_existing:
        existing = len(list(output_dir.glob("*.mid")))
        if existing > 0:
            print(f"  Found {existing:,} existing files (skipping)")
            total = total - existing
            print(f"  Remaining to download: {total:,}")

    print("-" * 70)

    # Stage 2: Download
    print("\n[Stage 2/3] Downloading MIDI files...")
    stage_start = time.time()
    sys.stdout.flush()

    ds = load_dataset("Metacreation/GigaMIDI", "v2.0.0", split="test", streaming=True)

    downloaded = 0
    skipped = 0
    total_size = 0

    with tqdm(
        total=total,
        desc="Downloading",
        unit="files",
        unit_scale=True,
        unit_divisor=1000,
    ) as pbar:
        for row in ds:
            md5 = row.get("md5", "")

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

            if downloaded % 1000 == 0 or downloaded == total:
                elapsed = time.time() - stage_start
                rate = downloaded / elapsed if elapsed > 0 else 0
                remaining = total - downloaded
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

    download_time = time.time() - stage_start
    print(f"\n  Downloaded: {downloaded:,} files in {download_time:.1f}s")
    print(f"  Skipped:   {skipped:,} files")
    print(f"  Speed:    {downloaded / download_time:.0f} files/s")
    sys.stdout.flush()

    # Stage 3: Verify
    print("\n[Stage 3/3] Verifying downloads...")
    stage_start = time.time()

    actual_files = list(output_dir.glob("*.mid"))
    verified = len(actual_files)
    total_verify_size = sum(f.stat().st_size for f in actual_files)

    print(f"  Verified: {verified:,} files")
    print(f"  Size:   {total_verify_size / (1024**3):.2f} GB")
    sys.stdout.flush()

    # Summary
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("COMPLETE - Summary")
    print("=" * 70)
    print(f"  Files downloaded: {downloaded:,}")
    print(f"  Files skipped:     {skipped:,}")
    print(f"  Files verified:    {verified:,}")
    print(f"  Total size:       {total_verify_size / (1024**3):.2f} GB")
    print(f"  Output:          {output_dir}")
    print(f"  Total time:       {total_time:.1f}s ({total_time / 60:.1f} min)")
    print("=" * 70)


if __name__ == "__main__":
    main()
