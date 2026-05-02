#!/usr/bin/env python3
"""
GigaMIDI Subset S4: All Expressive Files

This script creates S4 - ALL files with at least one expressive track (NOMML >= 12).

S4 Definition (from docs/amt-fine-tuning.md):
- Subset ID: S4
- Description: Expressive-all (all files with NOMML >= 12)
- Files: ~440k (31% of dataset)
- Selection Method: NOMML >= 12 filter

Usage:
    python scripts/gigamidi_create_s4.py [--nomml_threshold 12] [--output data/gigamidi_s4_expressive.json]
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Create S4: All Expressive Files (NOMML >= 12)"
    )
    parser.add_argument(
        "--nomml_threshold",
        type=int,
        default=12,
        help="NOMML threshold (default: 12)",
    )
    parser.add_argument(
        "--min_tracks",
        type=int,
        default=1,
        help="Minimum tracks (default: 1)",
    )
    parser.add_argument(
        "--max_tracks",
        type=int,
        default=16,
        help="Maximum tracks (default: 16)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/gigamidi_s4_all_expressive.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    sys.stdout.reconfigure(line_buffering=False)

    print("=" * 60)
    print("GigaMIDI Subset S4 Creator")
    print("All Expressive Files (NOMML >= 12)")
    print("=" * 60)
    print(f"NOMML threshold: >= {args.nomml_threshold}")
    print(f"Tracks: {args.min_tracks} - {args.max_tracks}")
    print("-" * 60)
    sys.stdout.flush()

    # Load dataset
    print("Loading GigaMIDI train split...")
    ds = load_dataset("Metacreation/GigaMIDI", "v2.0.0", split="train", streaming=True)

    # Collect expressive files
    print("Filtering for expressive tracks...")
    expressive = []
    total_counted = 0

    pbar = tqdm(total=1045726, desc="Filtering", unit="files", unit_scale=True)
    for row in ds:
        total_counted += 1
        pbar.update(1)

        nomml = row.get("NOMML", []) or []
        num_tracks = row.get("num_tracks", 0) or 0

        if num_tracks < args.min_tracks or num_tracks > args.max_tracks:
            continue

        has_expressive = any(n >= args.nomml_threshold for n in nomml)
        if not has_expressive:
            continue

        expressive.append(
            {
                "md5": row.get("md5"),
                "nomml": nomml,
                "num_tracks": num_tracks,
                "title": row.get("title", ""),
                "artist": row.get("artist", ""),
                "styles": row.get("music_styles_curated", []),
            }
        )

        if total_counted % 100000 == 0:
            pbar.set_postfix({"kept": len(expressive)})
            pbar.refresh()
    pbar.close()

    print(f"\nFiltered files: {len(expressive):,}")
    print(f"Match rate: {100 * len(expressive) / total_counted:.1f}%")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(expressive, f, indent=2)

    print(f"Saved to: {output_path}")
    print("=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
