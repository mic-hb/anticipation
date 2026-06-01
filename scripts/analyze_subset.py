#!/usr/bin/env python3
"""
Unified GigaMIDI subset analysis & calibration.
Supports 3 strategies with --strategy flag:
  B   — Big3-equalized (original Option B, calibrated)
  C1  — Uniform tracks across all instrument groups
  C2  — Equal % of each group's available tracks

Calibrates to a target % of Lakh MIDI tokens via binary search.

Usage:
    # Strategy B (Big3-equalized, calibrated to 50% LMD)
    uv run python scripts/analyze_subset.py --strategy B

    # Strategy C1 (uniform, 60% of Ethnic cap, calibrated to 50% LMD)
    uv run python scripts/analyze_subset.py --strategy C1 --pct 60

    # Strategy C2 (5% of each group, calibrated to 50% LMD)
    uv run python scripts/analyze_subset.py --strategy C2 --pct 5

    # Just show targets without calibration
    uv run python scripts/analyze_subset.py --strategy C1 --pct 60 --dry-run
"""

import argparse
import csv
import json
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


def compute_targets(strategy, pct, avail):
    """Compute per-group track targets for all discovered groups.
    Groups not in avail get cap=0 (prevent unbounded selection).
    """
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


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and calibrate GigaMIDI subsets"
    )
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["B", "C1", "C2"],
                        help="Subset strategy to analyze")
    parser.add_argument("--pct", type=float, default=None,
                        help="Parameter for strategy (C1: % of Ethnic cap, C2: % of each group)")
    parser.add_argument("--target", type=float, default=50.0,
                        help="Target %% of LMD tokens (default: 50)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show targets without running calibration/selection")
    args = parser.parse_args()

    if args.strategy in ("C1", "C2") and args.pct is None:
        parser.error(f"--strategy {args.strategy} requires --pct")

    csv_path = "Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv"
    if not Path(csv_path).exists():
        csv_path = f"/home/developer/auto-midi/lib/anticipation/{csv_path}"

    sys.stdout.reconfigure(line_buffering=True)
    csv.field_size_limit(1_000_000)
    start = time.time()

    target_tokens = LMD_EVENTS * args.target / 100
    strategy_label = f"{args.strategy}"
    if args.pct is not None:
        strategy_label += f" (pct={args.pct})"

    print(SEP)
    print(f"  SUBSET ANALYSIS — Strategy {strategy_label}")
    print(f"  Target: {args.target}% of LMD = {int(target_tokens):,} tokens")
    print(SEP)

    # ── Phase 1: Scan CSV ──────────────────────────────────────────────
    print("\nPhase 1: Scanning CSV...")
    group_available = Counter()
    other_files = []
    big3_only_by_cat = defaultdict(list)
    md5_notes = {}
    total_exp_files = 0

    nomml_by_group = defaultdict(list)
    notes_by_group = defaultdict(list)
    all_nomml = []
    all_notes = []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

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

        for j, gi in enumerate(ig):
            if j < len(nd_idx) and nomml[nd_idx[j]] >= 12:
                nomml_by_group[gi].append(nomml[nd_idx[j]])
                notes_by_group[gi].append(total_notes_row)

        all_nomml.extend(n for n in nomml if n != -1)
        all_notes.append(total_notes_row)

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
    print(f"\nPhase 2: Targets computed — {total_base:,} total tracks")

    if args.dry_run:
        print(f"\n{'─' * 50}")
        print("  DRY RUN — Targets only")
        print(f"{'─' * 50}")
        print(f"  {'Group':<23s} {'Target':>8s} {'Avail':>8s} {'%Avail':>6s}")
        print(f"  {'-'*47}")
        for gi in sorted(base_targets.keys(), key=lambda g: -base_targets[g]):
            t = base_targets[gi]
            a = avail[gi]
            ap = t / a * 100 if a else 0
            print(f"  {gi:<23s} {t:>8,d} {a:>8,d} {ap:>5.1f}%")
        print(f"\n  Run without --dry-run to calibrate and profile.")
        return

    # ── Phase 3: Binary search calibration ────────────────────────────
    print(f"\nPhase 3: Calibrating to {args.target}% LMD...")
    print(f"  Target: {int(target_tokens):,} tokens")
    print()

    lo, hi = 0.001, 20.0
    best_scale = 1.0
    best_error = float("inf")
    best_result = None

    for iteration in range(30):
        scale = (lo + hi) / 2
        tokens, tracks, c_files, groups, _, _ = evaluate(
            scale, base_targets, other_files, big3_only_by_cat, md5_notes)
        error = abs(tokens - target_tokens) / target_tokens
        pct_of_target = tokens / target_tokens * 100

        print(f"  iter {iteration:>2d}: scale={scale:.6f}  → {int(tokens):,} tok ({pct_of_target:.2f}%)"
              f"  err={error:.4f}  trk={tracks:,} files={c_files:,}")

        if error < best_error:
            best_error = error
            best_scale = scale
            best_result = (tokens, tracks, c_files, groups)

        if error < 0.005:
            print(f"  ✓ Converged")
            break

        if tokens > target_tokens:
            hi = scale
        else:
            lo = scale

    # Re-run with best scale to get full selection
    tokens, tracks, files, groups, all_selected, big3_cat_counts = evaluate(
        best_scale, base_targets, other_files, big3_only_by_cat, md5_notes)
    hours = tokens / EVENTS_PER_HOUR
    tokens_full_pct = tokens / LMD_EVENTS * 100

    train = sum(1 for m, _ in all_selected if m[0].lower() in set("0123456789abcd"))
    valid = sum(1 for m, _ in all_selected if m[0].lower() in set("e"))
    test = sum(1 for m, _ in all_selected if m[0].lower() in set("f"))

    big3_final = sum(groups.get(g, 0) for g in BIG3)
    other_final = tracks - big3_final
    notes_total = int(tokens / TOKENS_PER_NOTE)

    # ── PART 1: SUBSET PROFILE ────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"  PART 1: SUBSET PROFILE — Strategy {strategy_label}")
    print(f"{SEP}")
    print(f"  Scale factor:        {best_scale:.4f}")
    print(f"  Files:               {files:>10,d}")
    print(f"  Expressive tracks:   {tracks:>10,d}")
    print(f"  Total notes:         {notes_total:>10,d}")
    print(f"  Estimated tokens:    {int(tokens):>10,d}")
    print(f"  Estimated hours:     {hours:>10,.1f}")
    print(f"  % of target ({args.target:.0f}% LMD):  {tokens/target_tokens*100:>9.2f}%")
    print(f"  % of full LMD:       {tokens_full_pct:>9.2f}%")

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
        is_capped = args.strategy == "B" and gi in BIG3
        tag = " [CAP]" if is_capped else ""
        print(f"    {gi:<23s} {tc:>8,d} {pct:>7.2f}%{tag} {fc:>8,d}")
    print(f"    {'-'*50}")
    print(f"    {'TOTAL':<23s} {tracks:>8,d} {100:>7.2f}% {files:>8,d}")

    big3_pct = big3_final / tracks * 100
    other_pct = other_final / tracks * 100
    print(f"\n  Big3 total:  {big3_final:>8,d} ({big3_pct:.1f}%)")
    print(f"  Other total: {other_final:>8,d} ({other_pct:.1f}%)")

    if args.strategy == "B":
        print(f"\n  Big3-only sampling passes:")
        for cat in BIG3_ONLY_CATS:
            sel = big3_cat_counts.get(cat, 0)
            avail_cat = len(big3_only_by_cat[cat])
            print(f"    {cat}: {sel:,} files selected from {avail_cat:,} available")

    # ── PART 2: NOMML / TOTAL_NOTES DEEP-DIVE ─────────────────────────
    print(f"\n{SEP}")
    print("  PART 2: NOMML / TOTAL_NOTES DISTRIBUTIONS (all expressive)")
    print(SEP)

    print(f"\n  A. NOMML distribution per instrument group:")
    h = f"  {'Group':<23s} {'Count':>7s} {'Mean':>7s} {'Median':>7s} {'Min':>5s} {'P25':>5s} {'P75':>5s} {'Max':>5s}"
    h += f" {'≥20':>5s} {'≥30':>5s}"
    print(h)
    print(f"  {'-'*78}")
    for gi in sorted(nomml_by_group.keys(), key=lambda g: -len(nomml_by_group[g])):
        vals = nomml_by_group[gi]
        n = len(vals)
        svals = sorted(vals)
        mean = sum(vals) / n
        median = svals[n // 2]
        p25 = svals[n // 4]
        p75 = svals[3 * n // 4]
        gte20 = sum(1 for v in vals if v >= 20) / n * 100
        gte30 = sum(1 for v in vals if v >= 30) / n * 100
        print(f"  {gi:<23s} {n:>7,d} {mean:>7.1f} {median:>7.0f} {min(vals):>5.0f} {p25:>5.0f} {p75:>5.0f} {max(vals):>5.0f} {gte20:>4.0f}% {gte30:>4.0f}%")

    svals = sorted(all_nomml)
    n_all = len(svals)
    print(f"\n  Overall NOMML:")
    print(f"    Count:  {n_all:,}")
    print(f"    Mean:   {sum(all_nomml) / n_all:.1f}")
    print(f"    Median: {svals[n_all//2]:.0f}")
    print(f"    P25:    {svals[n_all//4]:.0f}")
    print(f"    P75:    {svals[3*n_all//4]:.0f}")
    print(f"    Min:    {min(svals):.0f}")
    print(f"    Max:    {max(svals):.0f}")

    print(f"\n  B. total_notes distribution per instrument group:")
    print(f"  {'Group':<23s} {'Count':>7s} {'Mean':>8s} {'Median':>8s} {'Min':>6s} {'P25':>7s} {'P75':>7s} {'Max':>8s}")
    print(f"  {'-'*78}")
    for gi in sorted(notes_by_group.keys(), key=lambda g: -len(notes_by_group[g])):
        vals = notes_by_group[gi]
        nv = len(vals)
        sv = sorted(vals)
        mean = sum(vals) / nv if nv else 0
        median = sv[nv // 2]
        p25 = sv[nv // 4]
        p75 = sv[3 * nv // 4]
        print(f"  {gi:<23s} {nv:>7,d} {mean:>8.0f} {median:>8,d} {min(vals):>6,d} {p25:>7,d} {p75:>7,d} {max(vals):>8,d}")

    print(f"\n  C. NOMML value distribution (tracks):")
    bins = [(12, 14), (14, 16), (16, 18), (18, 20), (20, 25), (25, 30), (30, 40), (40, 60), (60, 999)]
    ttn = len(all_nomml)
    for lo, hi in bins:
        cnt = sum(1 for v in all_nomml if lo <= v < hi)
        print(f"    [{lo:>3d}, {hi:>3d}): {cnt:>10,d} ({cnt/ttn*100:>5.1f}%)")
    cnt_ge_20 = sum(1 for v in all_nomml if v >= 20)
    print(f"    NOMML >= 20:     {cnt_ge_20:>10,d} ({cnt_ge_20/ttn*100:>5.1f}%)")

    print(f"\n  D. total_notes distribution (files):")
    nb = [(50, 100), (100, 200), (200, 500), (500, 1000), (1000, 2000),
          (2000, 5000), (5000, 10000), (10000, 9999999)]
    tfn = len(all_notes)
    for lo, hi in nb:
        cnt = sum(1 for v in all_notes if lo <= v < hi)
        print(f"    [{lo:>6,d}, {hi:>7,d}): {cnt:>10,d} ({cnt/tfn*100:>5.1f}%)")

    # ── PART 3: COMPARISON ────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PART 3: COMPARISON — This Subset vs Full Expressive vs LMD")
    print(SEP)

    full_exp_trk = sum(group_available.values())
    full_exp_n = sum(all_notes)
    full_exp_tok = full_exp_n * TOKENS_PER_NOTE
    full_exp_hr = full_exp_tok / EVENTS_PER_HOUR

    print(f"\n  A. Volume comparison")
    print(f"  {'-' * 62}")
    print(f"  {'Metric':<30s} {'Lakh MIDI':>10s} {'Full Exp':>10s} {'This':>10s}")
    print(f"  {'-'*62}")
    for name, lmd_v, full_v, this_v in [
        ("Files", LMD_FILES, total_exp_files, files),
        ("Notes", LMD_NOTES, full_exp_n, notes_total),
        ("Tokens", LMD_EVENTS, int(full_exp_tok), int(tokens)),
        ("Hours", LMD_HOURS, full_exp_hr, hours),
    ]:
        print(f"  {name:<30s} {fmt(lmd_v):>10s} {fmt(full_v):>10s} {fmt(this_v):>10s}")

    print(f"\n  B. Ratios relative to Lakh MIDI")
    print(f"  {'-' * 62}")
    for name, lmd_v, full_v, this_v in [
        ("Files (% LMD)", LMD_FILES, total_exp_files, files),
        ("Tokens (% LMD)", LMD_EVENTS, int(full_exp_tok), int(tokens)),
        ("Hours (% LMD)", LMD_HOURS, full_exp_hr, hours),
    ]:
        print(f"  {name:<30s} {'---':>10s} {full_v/lmd_v*100:>9.1f}% {this_v/lmd_v*100:>9.1f}%")

    print(f"\n  C. Instrument group distribution comparison")
    print(f"  {'-' * 63}")
    print(f"  {'Group':<23s} {'Full Trks':>9s} {'Full %':>7s} {'This Trks':>9s} {'This %':>7s} {'Δ%':>6s}")
    print(f"  {'-'*63}")
    for gi in all_group_names:
        full_c = group_available.get(gi, 0)
        full_p = full_c / full_exp_trk * 100 if full_exp_trk else 0
        this_c = groups.get(gi, 0)
        this_p = this_c / tracks * 100 if tracks else 0
        delta = this_p - full_p
        print(f"  {gi:<23s} {full_c:>9,d} {full_p:>6.2f}% {this_c:>9,d} {this_p:>6.2f}% {delta:>+5.2f}%")

    print(f"\n  D. Big3 vs Other split")
    print(f"  {'-' * 59}")
    full_big3 = sum(group_available[g] for g in BIG3)
    full_other = full_exp_trk - full_big3
    print(f"  {'Split':<23s} {'Full Exp':>10s} {'%':>7s} {'This':>10s} {'%':>7s}")
    print(f"  {'-'*59}")
    print(f"  {'Big3':<23s} {fmt(full_big3):>10s} {full_big3/full_exp_trk*100:>6.1f}% {fmt(big3_final):>10s} {big3_final/tracks*100:>6.1f}%")
    print(f"  {'Other':<23s} {fmt(full_other):>10s} {full_other/full_exp_trk*100:>6.1f}% {fmt(other_final):>10s} {other_final/tracks*100:>6.1f}%")

    if args.strategy == "B":
        print(f"\n  E. Big3-only composition comparison")
        print(f"  {'-' * 49}")
        print(f"  {'Category':<23s} {'Full Exp':>9s} {'This':>9s}")
        print(f"  {'-' * 44}")
        for cat in BIG3_ONLY_CATS:
            full_n = len(big3_only_by_cat[cat])
            this_n = big3_cat_counts.get(cat, 0)
            pct_s = this_n / full_n * 100 if full_n else 0
            print(f"  {cat:<23s} {full_n:>9,d} {this_n:>9,d} ({pct_s:.1f}%)")

    elapsed = time.time() - start
    print(f"\n{SEP}")
    print(f"  Analysis complete in {elapsed:.1f}s")
    print(SEP)


if __name__ == "__main__":
    main()
