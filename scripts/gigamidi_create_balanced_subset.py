#!/usr/bin/env python3
"""
GigaMIDI Balanced Subset Creator — Strategies B, C1, C2

Runs the exact greedy selection algorithm from analyze_subset.py, then copies
selected files from local GigaMIDI folder into Lakh hex-folder structure.

Calibrates to a target % of Lakh MIDI tokens via binary search.

Usage:
    # Strategy B (Big3-equalized, calibrated to 50% LMD)
    uv run python scripts/gigamidi_create_balanced_subset.py \
        --strategy B \
        --output data/gigamidi_balanced_B/ \
        --local_path /path/to/Final_GigaMIDI_V2.0_Final/

    # Strategy C1 (uniform, 60% of Ethnic cap, calibrated to 50% LMD)
    uv run python scripts/gigamidi_create_balanced_subset.py \
        --strategy C1 --pct 60 \
        --output data/gigamidi_balanced_C1/ \
        --local_path /path/to/Final_GigaMIDI_V2.0_Final/

    # Strategy C2 (5% of each group, calibrated to 50% LMD)
    uv run python scripts/gigamidi_create_balanced_subset.py \
        --strategy C2 --pct 5 \
        --output data/gigamidi_balanced_C2/ \
        --local_path /path/to/Final_GigaMIDI_V2.0_Final/

    # Dry-run to preview size and distribution
    uv run python scripts/gigamidi_create_balanced_subset.py \
        --strategy C1 --pct 60 --dry-run

    # Without --local_path (streams from HuggingFace — slower)
    uv run python scripts/gigamidi_create_balanced_subset.py \
        --strategy B \
        --output data/gigamidi_balanced_B/
"""

import argparse
import csv
import gc
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

TOKENS_PER_NOTE = 5.0
BIG3 = {"Piano", "Drums", "Guitar"}
LMD_FILES = 178_165
LMD_EVENTS = 663_163_605
LMD_NOTES = int(LMD_EVENTS / TOKENS_PER_NOTE)
LMD_HOURS = 8943
EVENTS_PER_HOUR = LMD_EVENTS / LMD_HOURS

BIG3_ONLY_CATS = ["piano+guitar", "guitar-only", "piano-only", "drums-only"]

TRAIN_HASHES = set("0123456789abcd")
VALID_HASHES = set("e")
TEST_HASHES = set("f")

SEP = "\u2500" * 72


def fmt(n):
    return f"{int(n):,}"


def parse_nomml(raw):
    try:
        return json.loads(raw.replace("'", '"'))
    except (json.JSONDecodeError, ValueError):
        return []


def parse_inst_groups(raw):
    if not raw or raw in ("[]", ""):
        return []
    try:
        return json.loads(raw.replace("'", '"'))
    except (json.JSONDecodeError, ValueError, SyntaxError):
        return []


def get_hex_folder(md5):
    return md5[0].lower()


def compute_targets(strategy, pct, avail):
    all_groups = sorted(avail.keys())
    targets = {}
    if strategy == "B":
        for gi in all_groups:
            if gi in BIG3:
                targets[gi] = 11_944
            else:
                targets[gi] = int(avail.get(gi, 0) * 0.0368)
    elif strategy == "C1":
        ethnic_avail = avail.get("Ethnic", 0)
        cap = int(ethnic_avail * pct / 100)
        for gi in all_groups:
            targets[gi] = min(cap, avail.get(gi, 0))
    elif strategy == "C2":
        for gi in all_groups:
            targets[gi] = int(avail.get(gi, 0) * pct / 100)
    return targets


def evaluate(scale, base_targets, other_files, big3_only_by_cat, md5_notes):
    scaled = {g: max(1, int(base_targets[g] * scale)) for g in base_targets}

    other_selected = []
    other_counts = Counter()
    big3_from_other = Counter()

    for md5, fgroups, _ in other_files:
        would_exceed = False
        for gi, c in fgroups.items():
            cap = scaled.get(gi, 0)
            if gi in BIG3:
                current = other_counts.get(gi, 0) + big3_from_other.get(gi, 0)
            else:
                current = other_counts.get(gi, 0)
            if cap and current + c > cap:
                would_exceed = True
                break
        if would_exceed:
            continue
        other_selected.append((md5, fgroups))
        for gi, c in fgroups.items():
            if gi in BIG3:
                big3_from_other[gi] += c
            else:
                other_counts[gi] += c

    remaining = {g: max(0, scaled.get(g, 0) - big3_from_other.get(g, 0))
                 for g in BIG3}
    big3_selected = []
    big3_counts = Counter()
    big3_cat_counts = {}

    def try_add(candidates):
        for md5, fgroups, _ in candidates:
            would_exceed = False
            for gi, c in fgroups.items():
                if gi in BIG3 and big3_counts[gi] + c > remaining.get(gi, 0):
                    would_exceed = True
                    break
            if would_exceed:
                continue
            big3_selected.append((md5, fgroups))
            for gi, c in fgroups.items():
                if gi in BIG3:
                    big3_counts[gi] += c

    for cat in BIG3_ONLY_CATS:
        before = len(big3_selected)
        try_add(big3_only_by_cat[cat])
        big3_cat_counts[cat] = len(big3_selected) - before

    all_s = other_selected + big3_selected
    sum_notes = sum(md5_notes.get(m, 0) for m, _ in all_s)
    tokens = sum_notes * TOKENS_PER_NOTE
    final_groups = Counter()
    for _, fgroups in all_s:
        for gi, c in fgroups.items():
            final_groups[gi] += c
    return tokens, sum(final_groups.values()), len(all_s), final_groups, all_s, big3_cat_counts


def build_local_md5_map(local_path):
    local_path = local_path.resolve()
    categories = ["no-drums", "all-instruments-with-drums", "drums-only"]
    splits = ["training-*/", "validation-*/", "test-*/"]

    md5_map = {}
    seen = set()

    for split_wildcard in splits:
        for category in categories:
            pattern = f"**/{split_wildcard}{category}/*/*.mid"
            for fpath in local_path.glob(pattern):
                if "__MACOSX" in fpath.parts:
                    continue
                md5 = fpath.stem
                if md5.startswith("._"):
                    continue
                if len(md5) != 32 or not all(c in "0123456789abcdef" for c in md5.lower()):
                    continue
                if md5 not in seen:
                    seen.add(md5)
                    md5_map[md5] = fpath

    return md5_map


def scan_existing_output(output_path):
    existing = set()
    if not output_path.exists():
        return existing
    for hex_folder in output_path.iterdir():
        if not hex_folder.is_dir():
            continue
        for fpath in hex_folder.iterdir():
            if fpath.suffix == ".mid":
                existing.add(fpath.stem)
    return existing


def write_target_md5s(target_md5s, split_name, output_base, local_md5_map, existing_md5s):
    total = len(target_md5s)
    if total == 0:
        return 0, 0, 0, 0

    print(f"\n  [{split_name}] Processing {total:,} files...")

    written = 0
    errors = 0
    skipped_existing = 0
    missing = 0
    pbar_start = time.time()

    pbar = tqdm(total=total, desc=f"  {split_name}", leave=True)
    for md5_val in target_md5s:
        if md5_val in existing_md5s:
            skipped_existing += 1
            pbar.update(1)
            continue

        local_path = local_md5_map.get(md5_val)
        if local_path is None:
            missing += 1
            pbar.update(1)
            continue

        try:
            out_folder = output_base / get_hex_folder(md5_val)
            out_folder.mkdir(parents=True, exist_ok=True)
            out_path = out_folder / f"{md5_val}.mid"
            shutil.copy2(local_path, out_path)
            existing_md5s.discard(md5_val)
            written += 1
        except Exception:
            errors += 1

        elapsed = time.time() - pbar_start
        rate = (written + skipped_existing) / elapsed if elapsed > 0 else 0
        remaining = (total - written - skipped_existing) / rate if rate > 0 else 0
        pbar.set_postfix(
            written=written, skipped=skipped_existing, missing=missing,
            eta=f"{int(remaining)}s" if remaining else "\u2014",
        )
        pbar.update(1)

    pbar.close()
    return written, errors, skipped_existing, missing


def profile_selection(all_selected, target_tokens, group_available, total_exp_files, best_scale):
    md5_notes = {m: 0 for m, _ in all_selected}
    groups = Counter()
    for _, fgroups in all_selected:
        for gi, c in fgroups.items():
            groups[gi] += c

    tracks = sum(groups.values())
    files = len(all_selected)
    tokens = 0
    hours = 0
    tokens_full_pct = 0

    print(f"\n{SEP}")
    print("  OUTPUT PROFILE")
    print(f"{SEP}")
    print(f"  Scale factor:        {best_scale:.4f}")
    print(f"  Files:               {files:>10,d}")
    print(f"  Expressive tracks:   {tracks:>10,d}")
    print(f"  Estimated tokens:    --- (requires total_notes)")
    print(f"  % of target (50% LMD): ---")
    print(f"  % of full LMD:       ---")

    train = sum(1 for m, _ in all_selected if m[0].lower() in TRAIN_HASHES)
    valid = sum(1 for m, _ in all_selected if m[0].lower() in VALID_HASHES)
    test = sum(1 for m, _ in all_selected if m[0].lower() in TEST_HASHES)

    print(f"\n  Splits (Lakh hex convention):")
    print(f"    Train (0-d): {train:>10,d} ({train/files*100:>5.1f}%)")
    print(f"    Valid (e):   {valid:>10,d} ({valid/files*100:>5.1f}%)")
    print(f"    Test (f):    {test:>10,d} ({test/files*100:>5.1f}%)")

    group_file_counts = Counter()
    for _, gdict in all_selected:
        for gi in gdict:
            group_file_counts[gi] += 1

    print(f"\n  Per-group distribution:")
    print(f"    {'Group':<23s} {'Tracks':>8s} {'%Tracks':>8s} {'Files':>8s}")
    print(f"    {'-'*50}")
    for gi, tc in sorted(groups.items(), key=lambda x: -x[1]):
        pct = tc / tracks * 100
        fc = group_file_counts.get(gi, 0)
        print(f"    {gi:<23s} {tc:>8,d} {pct:>7.2f}% {fc:>8,d}")
    print(f"    {'-'*50}")
    print(f"    {'TOTAL':<23s} {tracks:>8,d} {100:>7.2f}% {files:>8,d}")

    big3_final = sum(groups.get(g, 0) for g in BIG3)
    other_final = tracks - big3_final
    print(f"\n  Big3 total:  {big3_final:>8,d} ({big3_final/tracks*100:.1f}%)")
    print(f"  Other total: {other_final:>8,d} ({other_final/tracks*100:.1f}%)")

    return tokens


def main():
    parser = argparse.ArgumentParser(
        description="Create GigaMIDI balanced subsets (B, C1, C2)"
    )
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["B", "C1", "C2"],
                        help="Subset strategy to use")
    parser.add_argument("--pct", type=float, default=None,
                        help="Parameter for strategy (C1: %% of Ethnic cap, C2: %% of each group)")
    parser.add_argument("--target", type=float, default=50.0,
                        help="Target %% of LMD tokens (default: 50)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory path")
    parser.add_argument("--local_path", type=str, default=None,
                        help="Path to extracted Final_GigaMIDI folder (local copy mode)")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv")
    parser.add_argument("--dry-run", action="store_true",
                        help="Scan and print statistics without copying any files")
    parser.add_argument("--limit", type=int, default=None,
                        help="For debugging: limit CSV rows processed")
    args = parser.parse_args()

    if args.strategy in ("C1", "C2") and args.pct is None:
        parser.error(f"--strategy {args.strategy} requires --pct")

    sys.stdout.reconfigure(line_buffering=True)
    csv.field_size_limit(1_000_000)
    start = time.time()

    csv_path = args.csv_path
    if not csv_path:
        csv_path = "Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv"
    if not Path(csv_path).exists():
        csv_path = f"/home/developer/auto-midi/lib/anticipation/{csv_path}"

    output_path = Path(args.output)
    if not args.dry_run:
        output_path.mkdir(parents=True, exist_ok=True)

    strategy_label = f"{args.strategy}"
    if args.pct is not None:
        strategy_label += f" (pct={args.pct})"
    target_tokens = LMD_EVENTS * args.target / 100

    print(SEP)
    print(f"  BALANCED SUBSET CREATOR — Strategy {strategy_label}")
    print(f"  Target: {args.target}% of LMD = {int(target_tokens):,} tokens")
    print(f"  Output: {output_path}")
    if args.local_path:
        print(f"  Local:  {args.local_path}")
    if args.dry_run:
        print("  *** DRY RUN — no files will be written ***")
    print(SEP)

    # ── Phase 1: Scan CSV ──────────────────────────────────────────────
    print("\nPhase 1: Scanning CSV...")
    group_available = Counter()
    other_files = []
    big3_only_by_cat = defaultdict(list)
    md5_notes = {}
    total_exp_files = 0

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.limit:
        rows = rows[:args.limit]

    for row in tqdm(rows, desc="  Processing", unit=" rows"):
        nomml = parse_nomml(row.get("NOMML", "[]"))
        total_notes_row = int(row.get("total_notes", "0") or 0)
        md5 = row.get("md5", "")

        if total_notes_row < 50 or not any(n >= 12 for n in nomml):
            continue

        ig = parse_inst_groups(row.get("instrument_group (expressive)", "[]"))
        if not ig:
            continue

        nd_idx = [j for j, n in enumerate(nomml) if n != -1]
        file_groups = Counter()
        has_other = False

        for j, gi in enumerate(ig):
            if j < len(nd_idx) and nomml[nd_idx[j]] >= 12:
                file_groups[gi] += 1
                if gi not in BIG3:
                    has_other = True

        if not file_groups:
            continue

        total_exp_files += 1
        for gi, c in file_groups.items():
            group_available[gi] += c

        if md5:
            md5_notes[md5] = total_notes_row
            if has_other:
                other_files.append((md5, dict(file_groups), total_notes_row))
            else:
                key = None
                has_p = "Piano" in file_groups
                has_g = "Guitar" in file_groups
                has_d = "Drums" in file_groups
                if has_p and has_g:
                    key = "piano+guitar"
                elif has_p:
                    key = "piano-only"
                elif has_g:
                    key = "guitar-only"
                elif has_d:
                    key = "drums-only"
                if key:
                    big3_only_by_cat[key].append((md5, dict(file_groups), total_notes_row))

    other_files.sort(key=lambda x: -x[2])
    for cat in BIG3_ONLY_CATS:
        big3_only_by_cat[cat].sort(key=lambda x: -x[2])

    all_group_names = sorted(group_available.keys())
    avail = {g: group_available.get(g, 0) for g in all_group_names}
    total_avail = sum(avail.values())
    print(f"  Expressive files: {total_exp_files:,}")
    print(f"  Expressive tracks available: {total_avail:,}")

    # ── Phase 2: Targets ──────────────────────────────────────────────
    base_targets = compute_targets(args.strategy, args.pct, avail)
    total_base = sum(base_targets.values())
    print(f"\nPhase 2: Targets computed \u2014 {total_base:,} total tracks")

    # ── Phase 3: Binary search calibration ────────────────────────────
    print(f"\nPhase 3: Calibrating to {args.target}% LMD...")
    print(f"  Target: {int(target_tokens):,} tokens")
    print()

    lo, hi = 0.001, 20.0
    best_scale = 1.0
    best_error = float("inf")

    for iteration in range(30):
        scale = (lo + hi) / 2
        tokens_val, tracks_val, files_val, _, _, _ = evaluate(
            scale, base_targets, other_files, big3_only_by_cat, md5_notes)
        error = abs(tokens_val - target_tokens) / target_tokens
        pct_of_target = tokens_val / target_tokens * 100

        print(f"  iter {iteration:>2d}: scale={scale:.6f}  \u2192 {int(tokens_val):,} tok ({pct_of_target:.2f}%)"
              f"  err={error:.4f}  trk={tracks_val:,} files={files_val:,}")

        if error < best_error:
            best_error = error
            best_scale = scale

        if error < 0.005:
            print(f"  \u2713 Converged")
            break

        if tokens_val > target_tokens:
            hi = scale
        else:
            lo = scale

    # Re-run with best scale to get full selection
    tokens_val, tracks_val, files_val, groups, all_selected, big3_cat_counts = evaluate(
        best_scale, base_targets, other_files, big3_only_by_cat, md5_notes)

    target_md5s = {m for m, _ in all_selected}
    print(f"\n  Selected {len(target_md5s):,} files at scale={best_scale:.4f}")
    print(f"  Estimated tokens: {int(tokens_val):,} ({tokens_val/target_tokens*100:.2f}% of target)")

    if args.dry_run:
        profile_selection(all_selected, target_tokens, group_available, total_exp_files, best_scale)
        elapsed = time.time() - start
        print(f"\n{SEP}")
        print(f"  DRY RUN COMPLETE in {elapsed:.1f}s \u2014 no files written")
        print(f"{SEP}")
        print(f"  To create the subset:")
        path_hint = args.local_path or "/path/to/Final_GigaMIDI_V2.0_Final/"
        print(f"    uv run python scripts/gigamidi_create_balanced_subset.py \\")
        print(f"      --strategy {args.strategy}" +
              (f" --pct {args.pct}" if args.pct is not None else "") + " \\")
        print(f"      --output {args.output} \\")
        print(f"      --local_path {path_hint}")
        return

    # ── Phase 4: Copy files ──────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Phase 4: Copying selected files to output")
    print(f"{SEP}")

    existing_md5s = scan_existing_output(output_path)
    print(f"  Already on disk: {len(existing_md5s):,} files")

    if args.local_path:
        local_base = Path(args.local_path)
        if not local_base.exists():
            print(f"  ERROR: Local path does not exist: {local_base}")
            return

        print(f"\n  Scanning local folder for available files...")
        scan_start = time.time()
        local_md5_map = build_local_md5_map(local_base)
        scan_time = time.time() - scan_start
        print(f"  Found {len(local_md5_map):,} MIDI files in {scan_time:.1f}s")

        locally_found = target_md5s & local_md5_map.keys()
        locally_missing = target_md5s - local_md5_map.keys()
        print(f"  Target files available locally: {len(locally_found):,}")
        if locally_missing:
            print(f"  Target files NOT found locally: {len(locally_missing):,} (will be skipped)")

        if not locally_found:
            print("  No matching files found locally. Exiting.")
            return

        total_written = 0
        total_errors = 0
        total_skipped = 0
        total_missing = 0

        for split_name in ["train", "validation", "test"]:
            if split_name == "train":
                targets_in_split = {m for m in locally_found if m[0].lower() in TRAIN_HASHES}
            elif split_name == "validation":
                targets_in_split = {m for m in locally_found if m[0].lower() in VALID_HASHES}
            else:
                targets_in_split = {m for m in locally_found if m[0].lower() in TEST_HASHES}

            if not targets_in_split:
                print(f"\n  [{split_name}] No files in this split \u2014 skipping")
                continue

            w, e, s, m = write_target_md5s(
                targets_in_split, split_name, output_path,
                local_md5_map, existing_md5s,
            )
            total_written += w
            total_errors += e
            total_skipped += s
            total_missing += m

    else:
        print("\n  No --local_path provided. Streaming from HuggingFace...")
        print("  (This is slower. Consider providing --local_path for local copy.)")
        from datasets import load_dataset

        total_written = 0
        total_errors = 0
        total_skipped = 0
        total_missing = 0

        for split_name in ["train", "validation", "test"]:
            if split_name == "train":
                targets_in_split = {m for m in target_md5s if m[0].lower() in TRAIN_HASHES}
            elif split_name == "validation":
                targets_in_split = {m for m in target_md5s if m[0].lower() in VALID_HASHES}
            else:
                targets_in_split = {m for m in target_md5s if m[0].lower() in TEST_HASHES}

            if not targets_in_split:
                print(f"\n  [{split_name}] No files in this split \u2014 skipping")
                continue

            print(f"\n  [{split_name}] Streaming {len(targets_in_split):,} targets from HuggingFace...")
            ds = load_dataset("Metacreation/GigaMIDI", "v2.0.0", split=split_name, streaming=True)

            written = 0
            errors = 0
            skipped = 0
            pbar = tqdm(desc=f"  {split_name}", total=len(targets_in_split), leave=True)

            for row in ds:
                md5_val = row.get("md5", "")
                if md5_val not in targets_in_split:
                    continue

                if md5_val in existing_md5s:
                    skipped += 1
                    pbar.update(1)
                    continue

                midi_bytes = row.get("music", b"")
                if not midi_bytes:
                    errors += 1
                    pbar.update(1)
                    continue

                try:
                    out_folder = output_path / get_hex_folder(md5_val)
                    out_folder.mkdir(parents=True, exist_ok=True)
                    out_path = out_folder / f"{md5_val}.mid"
                    with open(out_path, "wb") as f:
                        f.write(midi_bytes)
                    existing_md5s.discard(md5_val)
                    written += 1
                except Exception:
                    errors += 1

                pbar.set_postfix(written=written, skipped=skipped)
                pbar.update(1)

            pbar.close()
            del ds
            gc.collect()

            total_written += written
            total_errors += errors
            total_skipped += skipped

    elapsed = time.time() - start

    # ── Phase 5: Profile output ──────────────────────────────────────
    profile_selection(all_selected, target_tokens, group_available, total_exp_files, best_scale)

    train_count = sum(len(list(output_path.glob(f"{h}/*"))) for h in "0123456789abcd")
    valid_count = len(list(output_path.glob("e/*")))
    test_count = len(list(output_path.glob("f/*")))

    print(f"\n{SEP}")
    print("  COPY SUMMARY")
    print(f"{SEP}")
    print(f"  Files written:       {total_written:>10,d}")
    print(f"  Already on disk:     {total_skipped:>10,d} (skipped)")
    print(f"  Write errors:        {total_errors:>10,d}")
    print(f"  Not found locally:   {total_missing:>10,d}")
    print(f"  {'─' * 42}")
    print(f"  Train (0-d):         {train_count:>10,d}")
    print(f"  Valid (e):           {valid_count:>10,d}")
    print(f"  Test (f):            {test_count:>10,d}")
    print(f"  {'─' * 42}")
    print(f"  Total on disk:       {train_count + valid_count + test_count:>10,d}")
    print(f"  Time:                {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output:              {output_path}")
    print(f"{SEP}")

    if total_missing > 0:
        print(f"\n  Note: {total_missing:,} target files were not found locally.")
        print(f"  Run with HuggingFace streaming (no --local_path) to fetch them.")

    print("\nNext steps:")
    print("  1. Preprocess: python gigamidi_preprocess_to_compound.py --input <output>/")
    print("  2. Tokenize:   python gigamidi_tokenize_events.py --input <output>/")
    print("  3. Define:    python gigamidi_define_splits.py --input <output>/")
    print("  4. Shuffle:   python gigamidi_shuffle_train.py --input <output>/")


if __name__ == "__main__":
    main()
