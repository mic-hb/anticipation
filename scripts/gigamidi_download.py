#!/usr/bin/env python3
"""
GigaMIDI Download Script

Downloads MIDI files from GigaMIDI dataset based on md5 list from subset creation.

Usage:
    python scripts/gigamidi_download.py --input data/gigamidi_s1_10pct_random.json --output data/gigamidi_s1_random/
"""

import argparse
import json
import os
import sys
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Download MIDI files from GigaMIDI using md5 list"
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

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=False)

    print("=" * 60)
    print("GigaMIDI Download Script")
    print("=" * 60)

    # Load md5 list
    print(f"Loading md5 list from: {args.input}")
    with open(args.input) as f:
        file_list = json.load(f)

    target_md5s = {item["md5"] for item in file_list}
    print(f"Target files: {len(target_md5s):,}")
    print(f"Output directory: {args.output}")
    print("-" * 60)
    sys.stdout.flush()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download from GigaMIDI
    print(f"Loading GigaMIDI '{args.split}' split...")
    ds = load_dataset(
        "Metacreation/GigaMIDI", "v2.0.0", split=args.split, streaming=True
    )

    downloaded = 0
    total_target = len(target_md5s)

    print(f"Downloading MIDI files...")

    for row in ds:
        md5 = row.get("md5", "")

        # Skip if not in our target list
        if md5 not in target_md5s:
            continue

        # Get binary MIDI data
        music_data = row.get("music")
        if not music_data:
            continue

        # Extract bytes from the 'bytes' field
        midi_bytes = music_data.get("bytes")
        if not midi_bytes:
            continue

        # Save to file
        midi_path = output_dir / f"{md5}.mid"
        with open(midi_path, "wb") as f:
            f.write(midi_bytes)

        downloaded += 1

        if downloaded % 1000 == 0:
            print(f"  Downloaded: {downloaded:,} / {total_target:,}")

        if args.limit and downloaded >= args.limit:
            print(f"  Reached limit of {args.limit} files")
            break

        if downloaded >= total_target:
            break

    print("-" * 60)
    print(f"Download complete!")
    print(f"  Downloaded: {downloaded:,} files")
    print(f"  Output: {output_dir}")
    print(
        f"  Total size: {sum(f.stat().st_size for f in output_dir.glob('*.mid')) / (1024**3):.2f} GB"
    )
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
