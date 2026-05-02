#!/usr/bin/env python3
"""
GigaMIDI Filtering Script (v2.0.0)

Filters the GigaMIDI dataset (v2.0.0) to select expressive tracks for fine-tuning.
Optimized for streaming mode with progress tracking.

Usage:
    python scripts/filter_gigamidi.py [--nomml_threshold N] [--output OUTPUT]
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Filter GigaMIDI dataset (v2.0.0) for expressive tracks"
    )
    parser.add_argument(
        "--nomml_threshold",
        type=int,
        default=12,
        help="Minimum NOMML score for expressive tracks (default: 12)",
    )
    parser.add_argument(
        "--min_tracks",
        type=int,
        default=1,
        help="Minimum number of tracks in file (default: 1)",
    )
    parser.add_argument(
        "--max_tracks",
        type=int,
        default=16,
        help="Maximum number of tracks in file (default: 16)",
    )
    parser.add_argument(
        "--styles",
        type=str,
        nargs="+",
        default=None,
        help="Filter by specific styles (e.g., classical jazz rock)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gigamidi_filtered.json",
        help="Output JSON file path (default: data/gigamidi_filtered.json)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Which split to process (default: train)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=False)

    n_total = {"train": 1045726, "validation": 130685, "test": 130712}

    print("=" * 60)
    print("GigaMIDI Filtering Script v2.0.0")
    print("=" * 60)
    print(f"Split: {args.split}")
    print(f"NOMML threshold: >= {args.nomml_threshold}")
    print(f"Tracks: {args.min_tracks} - {args.max_tracks}")
    if args.styles:
        print(f"Styles: {args.styles}")
    if args.limit:
        print(f"Limit: {args.limit} files")
    print("-" * 60)
    sys.stdout.flush()

    # Load in streaming mode
    print("Loading GigaMIDI dataset (streaming)...")
    sys.stdout.flush()

    ds = load_dataset(
        "Metacreation/GigaMIDI", "v2.0.0", split=args.split, streaming=True
    )

    total_files = args.limit or n_total[args.split]

    print(f"Processing ~{total_files:,} files...")
    sys.stdout.flush()

    filtered_metadata = []
    total_counted = 0
    total_no_expressive = 0

    # Progress bar
    pbar = tqdm(total=total_files, desc="Filtering", unit="files", unit_scale=True)

    for row in ds:
        total_counted += 1
        pbar.update(1)

        if total_counted % 100000 == 0:
            pbar.set_postfix(
                {"kept": len(filtered_metadata), "skip": total_no_expressive}
            )
            pbar.refresh()

        nomml = row.get("NOMML", []) or []
        num_tracks = row.get("num_tracks", 0) or 0

        # Track count filter
        if num_tracks < args.min_tracks or num_tracks > args.max_tracks:
            continue

        # NOMML filter
        has_expressive = any(n >= args.nomml_threshold for n in nomml)
        if not has_expressive:
            total_no_expressive += 1
            continue

        # Style filter
        if args.styles:
            styles = row.get("music_styles_curated", []) or []
            if not any(s in args.styles for s in styles):
                continue

        md5 = row.get("md5", "")
        if md5:
            filtered_metadata.append(
                {
                    "md5": md5,
                    "nomml": nomml,
                    "num_tracks": num_tracks,
                    "styles": row.get("music_styles_curated", []),
                    "title": row.get("title", ""),
                    "artist": row.get("artist", ""),
                }
            )

    pbar.close()

    print("-" * 60)
    print(f"Complete!")
    print(f"  Processed: {total_counted:,}")
    print(f"  No expressive: {total_no_expressive:,}")
    print(f"  KEPT (NOMML >= {args.nomml_threshold}): {len(filtered_metadata):,}")
    print("-" * 60)

    if not filtered_metadata:
        print("WARNING: No files matched!")
        return

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving to: {output_path}")
    sys.stdout.flush()

    with open(output_path, "w") as f:
        json.dump(filtered_metadata, f, indent=2)

    # Stats
    track_counts = [m["num_tracks"] for m in filtered_metadata]
    print(f"\nStats:")
    print(f"  Files: {len(filtered_metadata):,}")
    print(
        f"  Tracks/file: min={min(track_counts)}, max={max(track_counts)}, avg={sum(track_counts) / len(track_counts):.1f}"
    )
    print(f"  Rate: {100 * len(filtered_metadata) / total_counted:.1f}%")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
