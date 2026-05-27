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
import csv
import gc
import os
import random
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


TRAIN_HASHES = set("0123456789abcd")
VALID_HASHES = set("e")
TEST_HASHES = set("f")

# Full GigaMIDI split totals (from Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv)
# Derived from file_path prefix: training-V1.1-80%, validation-V1.1-10%, test-V1.1-10%
# NOTE: GigaMIDI does NOT use the Lakh hash-based split (0-d=train, e=valid, f=test).
# Used as tqdm total when --limit is not specified, enabling ETA for full runs.
FULL_SPLIT_COUNTS = {
    "train": 1_708_973,
    "validation": 213_621,
    "test": 213_624,
}


def get_hex_folder(md5: str) -> str:
    return md5[0].lower()


def scan_existing_output(output_path: Path) -> set[str]:
    """Walk output hex-folders and build set of md5s already on disk.

    One-time scan at startup: O(N) os calls, result is an in-memory set.
    Every subsequent row check becomes O(1) set lookup instead of O(I/O) stat().
    """
    existing = set()
    if not output_path.exists():
        return existing

    for hex_folder in output_path.iterdir():
        if not hex_folder.is_dir():
            continue
        for fpath in hex_folder.iterdir():
            if fpath.suffix == ".mid":
                # filename without .mid is the md5
                existing.add(fpath.stem)

    return existing


def _build_local_md5_map(local_path: Path) -> dict[str, Path]:
    """Recursively scan local GigaMIDI folder and build md5 → local_path map.

    Uses ** glob to skip any version folder name (e.g. V1.1, V2.0, etc.):
        {local_path}/**/training-*/{category}/{hex}/*.mid

    Skips __MACOSX metadata folders and invalid/non-md5 filenames.

    Checks instrument categories in order of completeness:
        no-drums → all-instruments-with-drums → drums-only

    Returns:
        dict[md5: str] -> Path of the local .mid file
    """
    # Ensure absolute path — Path.glob() with ** requires it
    local_path = local_path.resolve()

    categories = [
        "no-drums",
        "all-instruments-with-drums",
        "drums-only",
    ]
    splits = ["training-*/", "validation-*/", "test-*/"]

    md5_map = {}
    seen = set()

    for split_wildcard in splits:
        for category in categories:
            pattern = f"**/{split_wildcard}{category}/*/*.mid"
            for fpath in local_path.glob(pattern):
                # Skip __MACOSX metadata folders
                if "__MACOSX" in fpath.parts:
                    continue
                md5 = fpath.stem
                # Skip macOS metadata files (start with ._)
                if md5.startswith("._"):
                    continue
                # Validate md5: 32 hex chars
                if len(md5) != 32 or not all(c in "0123456789abcdef" for c in md5.lower()):
                    continue
                if md5 not in seen:
                    seen.add(md5)
                    md5_map[md5] = fpath

    return md5_map


def write_local_targets(target_md5s: set[str], split_name: str, output_base: Path,
                        existing_md5s: set[str]):
    """Copy target md5s from local GigaMIDI folder to output hex-folders.

    Uses the local folder scan map (built once). For each md5:
        - If already on disk (in existing_md5s): skip, O(1) lookup
        - If found locally: copy file, update existing_md5s
        - If not found locally: count as missing

    Returns:
        (written, errors, skipped_existing, missing) counts
    """
    total = len(target_md5s)
    print(f"\n  [{split_name}] Copying {total:,} files from local folder...")

    written = 0
    errors = 0
    skipped_existing = 0
    missing = 0
    pbar_start = time.time()

    pbar = tqdm(total=total, desc=f"  {split_name}", leave=True)
    for md5_val in target_md5s:
        # O(1) skip check — no disk I/O
        if existing_md5s is not None and md5_val in existing_md5s:
            skipped_existing += 1
            pbar.update(1)
            continue

        local_path = _local_md5_map.get(md5_val)
        if local_path is None:
            missing += 1
            pbar.update(1)
            continue

        try:
            out_folder = output_base / get_hex_folder(md5_val)
            out_folder.mkdir(parents=True, exist_ok=True)
            out_path = out_folder / f"{md5_val}.mid"
            shutil.copy2(local_path, out_path)
            if existing_md5s is not None:
                existing_md5s.discard(md5_val)  # mark as written
            written += 1
        except Exception:
            errors += 1

        elapsed = time.time() - pbar_start
        rate = (written + skipped_existing) / elapsed if elapsed > 0 else 0
        remaining = (total - written - skipped_existing) / rate if rate > 0 else 0
        pbar.set_postfix(
            written=written,
            skipped=skipped_existing,
            missing=missing,
            eta=f"{int(remaining)}s" if remaining else "—",
        )
        pbar.update(1)

    pbar.close()
    return written, errors, skipped_existing, missing


def _parse_python_list(val: str):
    """Parse a Python list literal like "['Reed', 'Bass']" or "[12, -1]" safely."""
    if not val or val in ("[]", ""):
        return []
    # Match integers or single-quoted strings
    strings = re.findall(r"'([^']*)'", val)
    if strings:
        return strings
    ints = [int(x) for x in re.findall(r"-?\d+", val)]
    return ints


def scan_csv_for_targets(csv_path: str, subset: str, nomml_threshold: int,
                         min_tracks: int, max_tracks: int,
                         sample_size, seed: int):
    """Scan the GigaMIDI metadata CSV and return the set of md5s to fetch.

    Reads CSV in one pass (chunked). Applies the same filter logic as the
    HuggingFace streaming path, but only touches lightweight metadata fields.
    The returned set is the EXACT list of rows whose music binary will be
    downloaded from HuggingFace (idempotent: already-downloaded files are
    skipped during write via write_midi_file).

    Returns:
        (target_md5s: set[str], split_counts: dict[str, int])
        target_md5s: set of md5 strings that pass the subset filter
        split_counts: {"train": N, "validation": N, "test": N} of matched rows
    """
    csv.field_size_limit(1_000_000)

    all_target_md5s = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="  CSV scan", unit="rows", leave=True):
            file_path = row.get("file_path", "")

            # Detect split from path prefix (GigaMIDI convention)
            if "training-V1.1-80%" in file_path:
                row_split = "train"
            elif "validation-V1.1-10%" in file_path:
                row_split = "validation"
            elif "test-V1.1-10%" in file_path:
                row_split = "test"
            else:
                continue  # unknown split

            md5_val = row.get("md5", "")
            if not md5_val:
                continue

            # --- Subset-specific filtering (same logic as filter_record) ---

            if subset == "s8":
                styles_curated = _parse_python_list(row.get("music_styles_curated", ""))
                scraped_raw = row.get("music_style_scraped", "") or ""
                scraped = _parse_python_list(scraped_raw)
                styles = set(s.lower() for s in styles_curated + scraped)
                if not any(s in styles for s in ("gospel", "latin")):
                    continue

            elif subset == "s11":
                styles_curated = _parse_python_list(row.get("music_styles_curated", ""))
                scraped_raw = row.get("music_style_scraped", "") or ""
                scraped = _parse_python_list(scraped_raw)
                styles = set(s.lower() for s in styles_curated + scraped)
                if not any(s in styles for s in ("gospel", "latin")):
                    continue
                nomml = _parse_python_list(row.get("NOMML", "[]"))
                if not any(n >= nomml_threshold for n in nomml):
                    continue

            elif subset in ("s2", "s4"):
                num_tracks = int(row.get("num_tracks", 0) or 0)
                if num_tracks < min_tracks or num_tracks > max_tracks:
                    continue
                nomml = _parse_python_list(row.get("NOMML", "[]"))
                if not any(n >= nomml_threshold for n in nomml):
                    continue

            # s1/s3: accept all (sampling handled below)

            all_target_md5s.append(md5_val)

    total = len(all_target_md5s)
    if total == 0:
        return set(), {"train": 0, "validation": 0, "test": 0}

    # Split counts for ETA (based on md5 leading char, Lakh hex folder convention)
    train = sum(1 for m in all_target_md5s if m[0].lower() in TRAIN_HASHES)
    valid = sum(1 for m in all_target_md5s if m[0].lower() in VALID_HASHES)
    test_ = sum(1 for m in all_target_md5s if m[0].lower() in TEST_HASHES)

    # For sampled subsets: random sample from the full filtered set
    if sample_size and sample_size < 1.0:
        random.seed(seed)
        sample_count = max(1, int(total * sample_size))
        sampled = set(random.sample(all_target_md5s, k=sample_count))
        print(f"  Sampled {sample_count:,} from {total:,} filtered ({sample_size*100:.1f}%)")
        # Recompute split counts for sampled set
        train = sum(1 for m in sampled if m[0].lower() in TRAIN_HASHES)
        valid = sum(1 for m in sampled if m[0].lower() in VALID_HASHES)
        test_ = sum(1 for m in sampled if m[0].lower() in TEST_HASHES)
        return sampled, {"train": train, "validation": valid, "test": test_}

    return set(all_target_md5s), {"train": train, "validation": valid, "test": test_}


def write_midi_file(midi_bytes, md5, output_base, existing_md5s=None):
    """Write a single MIDI file to hex folder (flat Lakh-style structure).

    Args:
        midi_bytes: raw MIDI binary
        md5: md5 string (used as filename)
        output_base: output root Path
        existing_md5s: if provided, O(1) in-memory set lookup replaces disk stat.
                       After successful write, md5 is removed from the set.

    Returns:
        (md5, split, skipped) — skipped is True when file was skipped (already in set)
    """
    # O(1) in-memory check when set is provided
    if existing_md5s is not None:
        if md5 in existing_md5s:
            split = (
                "train" if md5[0].lower() in TRAIN_HASHES else
                "valid" if md5[0].lower() in VALID_HASHES else
                "test"
            )
            return md5, split, True
        # Not in set — write, then add to set so subsequent rows skip it
        written = True
    else:
        # Fallback to disk check when no set provided
        midi_path = output_base / get_hex_folder(md5) / f"{md5}.mid"
        if midi_path.exists():
            split = (
                "train" if md5[0].lower() in TRAIN_HASHES else
                "valid" if md5[0].lower() in VALID_HASHES else
                "test"
            )
            return md5, split, True
        written = False

    folder = output_base / get_hex_folder(md5)
    folder.mkdir(parents=True, exist_ok=True)
    midi_path = folder / f"{md5}.mid"

    with open(midi_path, "wb") as f:
        f.write(midi_bytes)

    if existing_md5s is not None:
        existing_md5s.discard(md5)  # mark as written — skip if encountered again

    split = (
        "train" if md5[0].lower() in TRAIN_HASHES else
        "valid" if md5[0].lower() in VALID_HASHES else
        "test"
    )
    return md5, split, False


def write_filtered_targets(target_md5s: set[str], split_name: str, output_base: Path,
                            dry_run: bool, existing_md5s: set[str]):
    """Stream HuggingFace split, writing ONLY rows whose md5 is in target_md5s.

    CSV-guided mode: we already know the exact md5s to fetch. We iterate all
    rows (to maintain streaming), but ONLY access row["music"] for md5s in
    target_md5s. Non-matching rows: no binary download.

    existing_md5s: in-memory set of md5s already on disk. Checked before write
    to avoid disk I/O. Updated in-place after each write.

    Returns:
        (written, errors, skipped_existing) — matches stream_split_write signature
    """
    total_split = FULL_SPLIT_COUNTS.get(split_name, 0)
    print(f"\n  [{split_name}] Streaming all {total_split:,} rows, "
          f"{len(target_md5s):,} are S4 targets...")

    ds = load_dataset(
        "Metacreation/GigaMIDI",
        "v2.0.0",
        split=split_name,
        streaming=True,
    )

    written = 0
    errors = 0
    skipped_existing = 0
    total_seen = 0      # all rows scanned from this HuggingFace split
    total_matched = 0   # rows that were S4 targets (in target_md5s)
    pbar_start = time.time()

    # tqdm tracks all rows seen against full split size
    pbar = tqdm(total=total_split, desc=f"  {split_name}", leave=True)
    for row in ds:
        md5_val = row.get("md5", "")
        total_seen += 1
        pbar.update(1)

        if md5_val not in target_md5s:
            continue  # *** music NOT accessed — no binary download ***

        total_matched += 1

        midi_bytes = row.get("music", b"")
        if not midi_bytes:
            errors += 1
            continue

        _, _, skipped = write_midi_file(midi_bytes, md5_val, output_base, existing_md5s)
        if skipped:
            skipped_existing += 1
        else:
            written += 1

        elapsed = time.time() - pbar_start
        rate = total_seen / elapsed if elapsed > 0 else 0
        remaining = (total_split - total_seen) / rate if rate > 0 else 0
        pbar.set_postfix(
            matched=f"{total_matched}/{len(target_md5s)}",
            written=written,
            skipped=skipped_existing,
            eta=f"{int(remaining)}s" if remaining else "—",
        )

    pbar.close()
    del ds
    gc.collect()

    if dry_run:
        print(f"  [{split_name}] DRY RUN — would write {written:,} files")
        return 0, 0, 0
    return written, errors, skipped_existing


def filter_record(row, subset, nomml_threshold, min_tracks, max_tracks):
    """Return True if the record passes the subset filter, False otherwise."""
    md5_val = row.get("md5", "")
    if not md5_val:
        return False, None, None

    if subset == "s8":
        curated = row.get("music_styles_curated", []) or []
        scraped = row.get("music_style_scraped", "") or ""
        scraped = [scraped] if isinstance(scraped, str) else scraped
        styles = set(s.lower() for s in curated + scraped)
        if not any(s in styles for s in ("gospel", "latin")):
            return False, None, None

    elif subset == "s11":
        curated = row.get("music_styles_curated", []) or []
        scraped = row.get("music_style_scraped", "") or ""
        scraped = [scraped] if isinstance(scraped, str) else scraped
        styles = set(s.lower() for s in curated + scraped)
        if not any(s in styles for s in ("gospel", "latin")):
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
                      seed=42, dry_run=False, limit=None, existing_md5s=None):
    """
    Stream a single GigaMIDI split, filter, write immediately.
    For sampled subsets (s1/s3): writes all, then deletes oversample.

    existing_md5s: in-memory set of md5s already on disk (O(1) skip, no stat I/O).
                   After successful write, md5 is removed from the set.
    Returns (written, errors, sampled_and_kept, skipped_existing) counts.
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
        matched = 0
        scan_total = limit if limit else FULL_SPLIT_COUNTS.get(split_name, 0)
        pbar = tqdm(ds, desc=f"  Scan {split_name}", total=scan_total, leave=True)
        for row in pbar:
            if limit and scanned >= limit:
                break
            scanned += 1

            passed, md5_val, _ = filter_record(
                row, subset, nomml_threshold, min_tracks, max_tracks
            )
            if passed:
                all_accepted.append(md5_val)
                matched += 1
            pbar.set_postfix(matched=matched)

        print(f"  [{split_name}] Accepted {len(all_accepted):,} files (scanned {scanned:,})")

        if not all_accepted:
            return 0, 0, 0, 0

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
        skipped_existing = 0
        errors = 0
        pbar_start = time.time()

        for row in tqdm(ds, desc=f"  Write {split_name}", total=sample_count, leave=True):
            if row.get("md5", "") not in sampled_md5s:
                continue
            md5_val = row["md5"]
            midi_bytes = row.get("music", b"")
            if not midi_bytes:
                continue

            _, _, skipped = write_midi_file(midi_bytes, md5_val, output_base, existing_md5s)
            if skipped:
                skipped_existing += 1
            else:
                written += 1

            # Time estimation
            elapsed = time.time() - pbar_start
            rate = (written + skipped_existing) / elapsed if elapsed > 0 else 0
            remaining = (sample_count - written - skipped_existing) / rate if rate > 0 else 0
            pbar.set_postfix(written=written, skipped=skipped_existing,
                              eta=f"{int(remaining)}s" if remaining else "—")
        del ds
        gc.collect()
        return written, errors, sample_count, skipped_existing

    # Single-pass for non-sampled subsets (s4, s8, s11)
    print(f"  [{split_name}] Single-pass filter+write...")
    written = 0
    errors = 0
    skipped_existing = 0
    scanned = 0
    matched = 0
    pbar_start = time.time()

    # Use full split count as total when no limit set — enables ETA
    pbar_total = limit if limit else FULL_SPLIT_COUNTS.get(split_name, 0)
    pbar = tqdm(ds, desc=f"  {split_name}", total=pbar_total, leave=True)
    for row in pbar:
        if limit and scanned >= limit:
            break
        scanned += 1

        passed, md5_val, midi_bytes = filter_record(
            row, subset, nomml_threshold, min_tracks, max_tracks
        )
        if not passed:
            # Time estimation: scanned rate
            elapsed = time.time() - pbar_start
            rate = scanned / elapsed if elapsed > 0 else 0
            remaining = (pbar_total - scanned) / rate if rate > 0 else 0
            pbar.set_postfix(matched=matched, written=written, skipped=skipped_existing,
                              eta=f"{int(remaining)}s" if remaining else "—")
            continue

        matched += 1

        if dry_run:
            written += 1
            elapsed = time.time() - pbar_start
            rate = scanned / elapsed if elapsed > 0 else 0
            remaining = (pbar_total - scanned) / rate if rate > 0 else 0
            pbar.set_postfix(matched=matched, written=written, eta=f"{int(remaining)}s" if remaining else "—")
            continue

        # Write immediately (idempotent — skips existing via existing_md5s set)
        try:
            _, _, skipped = write_midi_file(midi_bytes, md5_val, output_base, existing_md5s)
            if skipped:
                skipped_existing += 1
            else:
                written += 1
        except Exception:
            errors += 1

        # Time estimation: matched rate as proxy for processing speed
        elapsed = time.time() - pbar_start
        rate = matched / elapsed if elapsed > 0 else 0
        remaining = (pbar_total - scanned) / rate if rate > 0 else 0
        pbar.set_postfix(matched=matched, written=written, skipped=skipped_existing,
                          eta=f"{int(remaining)}s" if remaining else "—")

    del ds
    gc.collect()
    if dry_run:
        print(f"  [{split_name}] DRY RUN — would write {written:,} files")
        return 0, 0, 0, 0
    return written, errors, 0, skipped_existing  # 0 = no sampling


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
    parser.add_argument(
        "--csv_path", type=str, default=None,
        help="Path to Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv. "
             "When provided, CSV is scanned first to build the exact set of md5s to fetch, "
             "then only matching rows download their music binary from HuggingFace. "
             "Much faster than pure streaming for subsets with high filter selectivity.",
    )
    parser.add_argument(
        "--local_path", type=str, default=None,
        help="Path to extracted Final_GigaMIDI_V* folder (e.g. Final_GigaMIDI_V2.0_Final/). "
             "When provided, files are copied from local disk instead of streamed from HuggingFace. "
             "Much faster. Uses wildcards to match any version naming. "
             "Can be combined with --csv_path to filter locally. "
             "Without --csv_path, copies all files found locally.",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Scan and print statistics without downloading or writing any files",
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
    elif not args.csv_path:
        print(f"Full run — ETA enabled via known split totals:")
        print(f"            train={FULL_SPLIT_COUNTS['train']:,}  "
              f"valid={FULL_SPLIT_COUNTS['validation']:,}  "
              f"test={FULL_SPLIT_COUNTS['test']:,}")
    print("-" * 70)

    if args.local_path:
        print(f"Local mode:   {args.local_path}")
        if args.csv_path:
            print(f"  + CSV:      {args.csv_path}")
            print(f"  → Filter local files via CSV, then copy from local disk (no HuggingFace)")
        else:
            print(f"  → Copy ALL local files to output (no HuggingFace, no CSV filter)")
    elif args.csv_path:
        print(f"CSV mode:     {args.csv_path}")
        print(f"  → CSV scan builds exact target md5 set, only matching rows fetch binary")
        print(f"  → Non-target rows: NO music download (zero network waste)")
    else:
        print("Streaming: one split at a time, immediate write, minimal RAM")

    if args.dry_run:
        print("*** DRY RUN MODE — no files will be written ***")
    print("=" * 70)

    total_written = 0
    total_errors = 0
    total_skipped = 0

    # Build in-memory set of md5s already on disk — O(1) skip, zero I/O per row
    existing_md5s = scan_existing_output(output_path)
    print(f"\n  Existing on disk: {len(existing_md5s):,} files")
    if existing_md5s and not args.dry_run:
        print(f"  → Will skip already-written files (idempotent, O(1) set lookup)")

    if args.local_path:
        # ---- Local folder mode: copy files directly from disk, no HuggingFace ----
        local_base = Path(args.local_path)
        if not local_base.exists():
            print(f"  ERROR: Local path does not exist: {local_base}")
            return

        print(f"\n[Local mode] Scanning {local_base} for .mid files...")
        scan_start = time.time()
        global _local_md5_map
        _local_md5_map = _build_local_md5_map(local_base)
        scan_time = time.time() - scan_start
        print(f"  → Scanned {len(_local_md5_map):,} files in {scan_time:.1f}s")

        # If CSV also provided, filter locally-copied files by subset
        if args.csv_path:
            print(f"\n  [CSV mode] Applying subset filter to local files...")
            target_md5s, split_counts = scan_csv_for_targets(
                args.csv_path,
                args.subset,
                args.nomml_threshold,
                args.min_tracks,
                args.max_tracks,
                args.sample_size,
                args.seed,
            )
            total_targets = len(target_md5s)
            # Only keep md5s that are both in CSV targets AND in local folder
            locally_found = target_md5s & _local_md5_map.keys()
            locally_missing = target_md5s - _local_md5_map.keys()
            print(f"  → Total CSV target md5s: {total_targets:,}  "
                  f"(train={split_counts['train']:,}  "
                  f"valid={split_counts['validation']:,}  "
                  f"test={split_counts['test']:,})")
            print(f"  → Found locally: {len(locally_found):,}  "
                  f"(will copy from local, no HuggingFace)")
            if locally_missing:
                print(f"  → Not found locally: {len(locally_missing):,}  "
                      f"(will skip — no HuggingFace fallback in local mode)")
            total_targets = len(locally_found)

            if total_targets == 0:
                print("  → No matching files found. Exiting.")
                return

            for split_name in ["train", "validation", "test"]:
                if split_name == "train":
                    targets_in_split = {m for m in locally_found if m[0].lower() in TRAIN_HASHES}
                elif split_name == "validation":
                    targets_in_split = {m for m in locally_found if m[0].lower() in VALID_HASHES}
                else:
                    targets_in_split = {m for m in locally_found if m[0].lower() in TEST_HASHES}

                if not targets_in_split:
                    print(f"\n  [{split_name}] No files in this split — skipping")
                    continue

                w, e, skipped, missing = write_local_targets(
                    targets_in_split,
                    split_name,
                    output_path,
                    existing_md5s,
                )
                total_written += w
                total_errors += e
                total_skipped += skipped
        else:
            # No CSV: copy ALL local files, partitioned by md5 leading char
            all_local_md5s = set(_local_md5_map.keys())
            print(f"  → Copying ALL {len(all_local_md5s):,} local files (no CSV filter)")

            for split_name in ["train", "validation", "test"]:
                if split_name == "train":
                    targets_in_split = {m for m in all_local_md5s if m[0].lower() in TRAIN_HASHES}
                elif split_name == "validation":
                    targets_in_split = {m for m in all_local_md5s if m[0].lower() in VALID_HASHES}
                else:
                    targets_in_split = {m for m in all_local_md5s if m[0].lower() in TEST_HASHES}

                if not targets_in_split:
                    continue

                w, e, skipped, missing = write_local_targets(
                    targets_in_split,
                    split_name,
                    output_path,
                    existing_md5s,
                )
                total_written += w
                total_errors += e
                total_skipped += skipped

    elif args.csv_path:
        # ---- CSV-guided HuggingFace streaming mode ----
        print("\n[CSV mode] Scanning metadata CSV to build target md5 set...")
        target_md5s, split_counts = scan_csv_for_targets(
            args.csv_path,
            args.subset,
            args.nomml_threshold,
            args.min_tracks,
            args.max_tracks,
            args.sample_size,
            args.seed,
        )
        total_targets = len(target_md5s)
        print(f"  → Total target md5s: {total_targets:,}  "
              f"(train={split_counts['train']:,}  "
              f"valid={split_counts['validation']:,}  "
              f"test={split_counts['test']:,})")

        if total_targets == 0:
            print("  → No matching files found. Exiting.")
            return

        for split_name in ["train", "validation", "test"]:
            # Partition target_md5s by split (Lakh hex folder convention)
            if split_name == "train":
                targets_in_split = {m for m in target_md5s if m[0].lower() in TRAIN_HASHES}
            elif split_name == "validation":
                targets_in_split = {m for m in target_md5s if m[0].lower() in VALID_HASHES}
            else:
                targets_in_split = {m for m in target_md5s if m[0].lower() in TEST_HASHES}

            if not targets_in_split:
                print(f"\n  [{split_name}] No files in this split — skipping")
                continue

            w, e, skipped = write_filtered_targets(
                targets_in_split,
                split_name,
                output_path,
                args.dry_run,
                existing_md5s,
            )
            total_written += w
            total_errors += e
            total_skipped += skipped

    else:
        # ---- Original streaming mode (filter on-the-fly) ----
        for split_name in ["train", "validation", "test"]:
            w, e, _, skipped = stream_split_write(
                split_name=split_name,
                subset=args.subset,
                output_base=output_path,
                nomml_threshold=args.nomml_threshold,
                min_tracks=args.min_tracks,
                max_tracks=args.max_tracks,
                workers=args.workers,
                sample_size=args.sample_size,
                seed=args.seed,
                dry_run=args.dry_run,
                limit=args.limit,
                existing_md5s=existing_md5s,
            )
            total_written += w
            total_errors += e
            total_skipped += skipped
    elapsed = time.time() - start_time

    if args.dry_run:
        print("\n" + "=" * 70)
        print("DRY RUN COMPLETE — no files were written")
        print("=" * 70)
        print(f"  Would write: {total_written:,} files")
        print(f"  Already on disk: {total_skipped:,} files (idempotent — will be skipped)")
        print(f"  Time:        {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        print("=" * 70)
        print("\nRun without --dry_run to download and write files.")
        return

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
    print(f"  Files written:      {total_written:,}")
    print(f"  Already on disk:     {total_skipped:,} (idempotent — skipped)")
    print(f"  Write errors:       {total_errors:,}")
    print(f"    Train (0-d):      {train_count:,}")
    print(f"    Valid (e):        {valid_count:,}")
    print(f"    Test (f):        {test_count:,}")
    print(f"  Output:             {output_path}")
    print(f"  Time:               {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 70)
    if total_skipped > 0:
        print(f"\n  Note: {total_skipped:,} files were skipped (already exist) — reruns are idempotent")
    print("\nNext steps:")
    print("  1. Preprocess: python gigamidi_preprocess_to_compound.py --input <output>/")
    print("  2. Tokenize:   python gigamidi_tokenize_events.py --input <output>/")
    print("  3. Define:    python gigamidi_define_splits.py --input <output>/")
    print("  4. Shuffle:   python gigamidi_shuffle_train.py --input <output>/")


if __name__ == "__main__":
    main()
