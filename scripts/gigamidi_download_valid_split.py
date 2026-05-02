#!/usr/bin/env python3
"""
GigaMIDI Validation Split Downloader

Downloads all MIDI files from the GigaMIDI validation split.

Usage:
    python scripts/gigamidi_download_valid.py --output data/gigamidi_valid_raw/

Output structure (flat):
    output_dir/
        0000123456789abcdef1234.mid
        abcd...
"""

import argparse
import sys
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Download GigaMIDI validation split")
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

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=False)

    print("=" * 60)
    print("GigaMIDI Validation Split Downloader")
    print("=" * 60)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading GigaMIDI validation split...")
    ds = load_dataset(
        "Metacreation/GigaMIDI", "v2.0.0", split="validation", streaming=True
    )

    # First pass: count
    print("Counting files...")
    total = 0
    for _ in ds:
        total += 1
    print(f"Total validation files: {total:,}")

    # Reload and download
    print(f"Downloading to {output_dir}...")
    ds = load_dataset(
        "Metacreation/GigaMIDI", "v2.0.0", split="validation", streaming=True
    )

    downloaded = 0
    for row in ds:
        md5 = row.get("md5", "")
        music_data = row.get("music")
        if not music_data:
            continue

        midi_bytes = music_data.get("bytes")
        if not midi_bytes:
            continue

        midi_path = output_dir / f"{md5}.mid"
        with open(midi_path, "wb") as f:
            f.write(midi_bytes)

        downloaded += 1

        if downloaded % 1000 == 0:
            print(f"  Downloaded: {downloaded:,} / {total:,}")

        if args.limit and downloaded >= args.limit:
            print(f"  Reached limit of {args.limit}")
            break

    print("-" * 60)
    print(f"Download complete!")
    print(f"  Downloaded: {downloaded:,} files")
    print(f"  Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
