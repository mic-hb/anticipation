#!/usr/bin/env python3
"""
Explore balanced subset strategies across all instrument groups.
Shows per-group track targets for each strategy. No token estimation
heuristics — token calibration done via actual binary search on chosen strategy.

Usage:
    uv run python scripts/explore_balanced_strategies.py
    uv run python scripts/explore_balanced_strategies.py --calibrate C3
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

TOKENS_PER_NOTE = 5.0
BIG3 = {"Piano", "Drums", "Guitar"}
LMD_EVENTS = 663_163_605
TARGET_TOKENS = LMD_EVENTS * 0.5

# Groups with >0 available tracks
ALL_GROUPS = [
    "Piano", "Drums", "Guitar", "Brass", "Chromatic Percussion",
    "Ensemble", "Ethnic", "Organ", "Percussive", "Pipe",
    "Reed", "Sound Effects", "Strings", "Synth Lead", "Synth Pad",
]
N_GROUPS = len(ALL_GROUPS)  # = 15

SEP = "=" * 72


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", type=str, default=None,
                        help="Run actual calibration on a strategy (e.g., C1, C3)")
    args = parser.parse_args()

    csv_path = "Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv"
    if not Path(csv_path).exists():
        csv_path = f"/home/developer/auto-midi/lib/anticipation/{csv_path}"

    sys.stdout.reconfigure(line_buffering=True)
    csv.field_size_limit(1_000_000)

    print(SEP)
    print("  EXPLORING BALANCED SUBSET STRATEGIES")
    print(SEP)

    # ── Scan CSV ──────────────────────────────────────────────────────
    print("\nScanning CSV for per-group stats...")
    group_available = Counter()

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in tqdm(rows, desc="  Processing"):
        nomml = parse_nomml(row.get("NOMML", "[]"))
        total_notes = int(row.get("total_notes", "0") or 0)

        if total_notes < 50 or not any(n >= 12 for n in nomml):
            continue

        ig = parse_inst_groups(row.get("instrument_group (expressive)", "[]"))
        if not ig:
            continue

        nd_idx = [j for j, n in enumerate(nomml) if n != -1]
        file_groups = Counter()

        for j, gi in enumerate(ig):
            if j < len(nd_idx) and nomml[nd_idx[j]] >= 12:
                file_groups[gi] += 1

        if not file_groups:
            continue

        for gi, c in file_groups.items():
            group_available[gi] += c

    # Filter to only existing groups
    avail = {g: group_available.get(g, 0) for g in ALL_GROUPS}
    total_avail = sum(avail.values())

    print(f"  Active groups: {N_GROUPS}")
    print(f"  Total expressive tracks available: {total_avail:,}")

    # ── Strategy definitions ──────────────────────────────────────────
    strategies = {}

    # Reference: current Option B
    strategies["B (current): Big3-equalized"] = {}
    for gi in ALL_GROUPS:
        if gi in BIG3:
            strategies["B (current): Big3-equalized"][gi] = 44_444
        else:
            strategies["B (current): Big3-equalized"][gi] = int(avail[gi] * 0.1368)

    strategies["B-calibrated: Big3-equalized (50% LMD)"] = {}
    for gi in ALL_GROUPS:
        if gi in BIG3:
            strategies["B-calibrated: Big3-equalized (50% LMD)"][gi] = 11_944
        else:
            strategies["B-calibrated: Big3-equalized (50% LMD)"][gi] = int(avail[gi] * 0.0368)

    # Strategy C1: Uniform — all 15 groups get same track count
    # Bottleneck is Ethnic = 5,163; scaled to ~4k for feasibility
    ethnic_avail = avail["Ethnic"]
    for cap_pct in [90, 80, 60, 50, 40, 30, 20]:
        capped = int(ethnic_avail * cap_pct / 100)
        strategies[f"C1: Uniform tracks ({cap_pct}%)"] = {}
        for gi in ALL_GROUPS:
            strategies[f"C1: Uniform tracks ({cap_pct}%)"][gi] = min(capped, avail[gi])

    # Strategy C2: Per-group % target (each gets exactly pct% of its available)
    for pct in [30, 20, 15, 10, 7, 5]:
        strategies[f"C2: {pct}% of each group's available"] = {}
        for gi in ALL_GROUPS:
            strategies[f"C2: {pct}% of each group's available"][gi] = int(avail[gi] * pct / 100)

    # Strategy C3: Sqrt-weighted — target % proportional to sqrt(current %)
    strategies["C3: Sqrt-weighted"] = {}
    sqrt_weights = {}
    sqrt_total = 0
    for gi in ALL_GROUPS:
        w = math.sqrt(avail[gi] / total_avail)
        sqrt_weights[gi] = w
        sqrt_total += w
    for gi in ALL_GROUPS:
        target_pct = sqrt_weights[gi] / sqrt_total
        strategies["C3: Sqrt-weighted"][gi] = int(total_avail * target_pct)

    # Strategy C4: Log-weighted — target % proportional to log(1 + current %)
    strategies["C4: Log-weighted"] = {}
    log_weights = {}
    log_total = 0
    for gi in ALL_GROUPS:
        pct = avail[gi] / total_avail * 100
        w = math.log(1 + pct)
        log_weights[gi] = w
        log_total += w
    for gi in ALL_GROUPS:
        target_pct = log_weights[gi] / log_total
        strategies["C4: Log-weighted"][gi] = int(total_avail * target_pct)

    # Strategy C5: Compressed range toward uniform
    strategies["C5: 25% compressed"] = {}
    uniform_pct = 100 / N_GROUPS
    for gi in ALL_GROUPS:
        pct = avail[gi] / total_avail * 100
        compressed = uniform_pct + (pct - uniform_pct) * 0.25
        strategies["C5: 25% compressed"][gi] = int(total_avail * compressed / 100)

    # Strategy C6: Proportional with floor — each gets at least MIN tracks
    for floor_pct in [5, 4, 3, 2]:
        strategies[f"C6: Floor {floor_pct}% of total, rest proportional"] = {}
        floor_tracks = int(total_avail * floor_pct / 100)
        floor_reserved = floor_tracks * N_GROUPS
        if floor_reserved > total_avail:
            continue
        remaining = total_avail - floor_reserved
        avail_no_max = sum(avail[gi] for gi in ALL_GROUPS)
        for gi in ALL_GROUPS:
            base = floor_tracks
            extra = int(remaining * avail[gi] / avail_no_max)
            strategies[f"C6: Floor {floor_pct}% of total, rest proportional"][gi] = min(base + extra, avail[gi])

    # ── Evaluate all strategies ──────────────────────────────────────
    print(f"\n{SEP}")
    print("  STRATEGY COMPARISON — Track Targets")
    print(SEP)

    strat_summary = []
    for sname, targets in strategies.items():
        total_tracks = sum(targets.values())
        # Check feasibility
        feasible = True
        for gi in ALL_GROUPS:
            if targets[gi] > avail[gi]:
                feasible = False
                break
        if not feasible:
            continue

        # Per-group percentages
        max_pct = max(targets[g] / total_tracks * 100 for g in targets)
        min_pct = min(targets[g] / total_tracks * 100 for g in targets)
        top3 = [(g, targets[g]) for g in sorted(targets.keys(), key=lambda g: -targets[g])[:3]]
        bottom3 = [(g, targets[g]) for g in sorted(targets.keys(), key=lambda g: targets[g])[:3]]

        strat_summary.append((sname, total_tracks, max_pct, min_pct, top3, bottom3, targets))

    # Sort by max-min range (more balanced = smaller range first)
    strat_summary.sort(key=lambda r: r[2] - r[3])

    print(f"\n  {'Strategy':<42s} {'Tracks':>8s} {'Max%':>7s} {'Min%':>7s} {'Range':>7s}")
    print(f"  {'-'*73}")
    for sname, tracks, max_pct, min_pct, top3, bottom3, _ in strat_summary:
        rng = max_pct - min_pct
        print(f"  {sname:<42s} {tracks:>8,d} {max_pct:>6.2f}% {min_pct:>6.2f}% {rng:>6.2f}%")

    # ── Detailed view ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  DETAILED VIEW — All Feasible Strategies")
    print(SEP)

    for sname, tracks, max_pct, min_pct, top3, bottom3, targets in strat_summary:
        print(f"\n  {'─' * 55}")
        print(f"  ▶ {sname}")
        print(f"  {'─' * 55}")
        print(f"  Total tracks: {tracks:,}")
        print(f"  Max group: {max_pct:.1f}% ({top3[0][0]}={top3[0][1]:,})  "
              f"Min group: {min_pct:.1f}% ({bottom3[0][0]}={bottom3[0][1]:,})")
        print(f"  Top 3: {', '.join(f'{g}={t:,}' for g, t in top3)}")
        print(f"  Bottom 3: {', '.join(f'{g}={t:,}' for g, t in bottom3)}")
        print()
        print(f"  {'Group':<23s} {'Target':>8s} {'%Total':>7s} {'Avail':>8s} {'%Avail':>6s}")
        print(f"  {'-'*54}")
        for gi in sorted(targets.keys(), key=lambda g: -targets[g]):
            t = targets[gi]
            a = avail[gi]
            p = t / tracks * 100
            ap = t / a * 100 if a else 0
            print(f"  {gi:<23s} {t:>8,d} {p:>6.2f}% {a:>8,d} {ap:>5.1f}%")
        print()

    # ── Calibrate selected strategy ──────────────────────────────────
    if args.calibrate:
        prefix = args.calibrate.upper()
        matches = [s for s in strat_summary if s[0].startswith(prefix)]
        if not matches:
            print(f"  No strategy matching '{prefix}'. Available: {[s[0] for s in strat_summary]}")
            return

        sname, tracks, max_pct, min_pct, top3, bottom3, targets = matches[0]
        print(f"\n{SEP}")
        print(f"  CALIBRATING: {sname}")
        print(SEP)
        print(f"  Running actual greedy selection with binary search...")
        print()

        # Re-run with actual calibration
        calibrate_strategy(targets, avail)


def calibrate_strategy(fixed_targets, avail):
    """Binary search a scale factor applied to all targets to hit 50% LMD."""
    # Extract targets that are within available
    base_targets = {}
    for gi, t in fixed_targets.items():
        if t <= avail[gi]:
            base_targets[gi] = t
        elif avail[gi] > 0:
            base_targets[gi] = avail[gi]

    total_base = sum(base_targets.values())

    # We need to scan the CSV and run the selection for each scale factor.
    # Load the relevant data structures first.
    csv_path = "Final-Metadata-Extended-GigaMIDI-Dataset-updated.csv"
    if not Path(csv_path).exists():
        csv_path = f"/home/developer/auto-midi/lib/anticipation/{csv_path}"

    print("  Loading file-level data...")
    other_files = []
    big3_only_by_cat = defaultdict(list)
    md5_notes = {}
    BIG3_ONLY_CATS = ["piano+guitar", "guitar-only", "piano-only", "drums-only"]
    BIG3 = {"Piano", "Drums", "Guitar"}

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for row in tqdm(rows, desc="  Processing"):
        nomml = parse_nomml(row.get("NOMML", "[]"))
        total_notes_v = int(row.get("total_notes", "0") or 0)
        md5 = row.get("md5", "")

        if total_notes_v < 50 or not any(n >= 12 for n in nomml):
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

        if md5:
            md5_notes[md5] = total_notes_v
            if has_other:
                other_files.append((md5, dict(file_groups), total_notes_v))
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
                    big3_only_by_cat[key].append((md5, dict(file_groups), total_notes_v))

    # Sort by total_notes descending (largest files first)
    other_files.sort(key=lambda x: -x[2])
    for cat in BIG3_ONLY_CATS:
        big3_only_by_cat[cat].sort(key=lambda x: -x[2])

    def evaluate(scale):
        scaled_targets = {g: max(1, int(base_targets[g] * scale)) for g in base_targets}
        big3_cap = max(scaled_targets.get(g, 0) for g in BIG3)
        # Actually for each strategy, we need specific per-group caps
        group_caps = scaled_targets

        other_selected = []
        other_counts = Counter()
        big3_from_other = Counter()

        for md5, fgroups, _ in other_files:
            would_exceed = False
            for gi, c in fgroups.items():
                cap = group_caps.get(gi, 0)
                # For Big3 groups in other files, Big3 tracks might overflow
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

        # Big3-only files: allocate remaining caps
        remaining = {}
        for gi in BIG3:
            from_other = big3_from_other.get(gi, 0)
            remaining[gi] = max(0, group_caps.get(gi, 0) - from_other)
        big3_selected = []
        big3_counts = Counter()

        def try_add(candidates):
            added = 0
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
                added += 1
            return added

        for cat in BIG3_ONLY_CATS:
            try_add(big3_only_by_cat[cat])

        all_selected = other_selected + big3_selected
        sum_notes = sum(md5_notes.get(m, 0) for m, _ in all_selected)
        tokens = sum_notes * TOKENS_PER_NOTE
        final_tracks = sum(1 for _, g in all_selected for _ in g)
        # Better: sum groups
        final_groups = Counter()
        for _, fgroups in all_selected:
            for gi, c in fgroups.items():
                final_groups[gi] += c
        return tokens, sum(final_groups.values()), len(all_selected), final_groups

    print(f"  Target: {int(TARGET_TOKENS):,} tokens")
    print(f"  Tolerance: ±1%")

    lo, hi = 0.01, 10.0
    best_scale = 1.0
    best_error = float("inf")
    best_result = None

    for iteration in range(30):
        scale = (lo + hi) / 2
        tokens, tracks, files, groups = evaluate(scale)
        error = abs(tokens - TARGET_TOKENS) / TARGET_TOKENS
        pct_of_target = tokens / TARGET_TOKENS * 100

        print(f"  iter {iteration:>2d}: scale={scale:.4f}  → {int(tokens):,} tok ({pct_of_target:.1f}%)"
              f"  err={error:.4f}  tracks={tracks:,} files={files:,}")

        if error < best_error:
            best_error = error
            best_scale = scale
            best_result = (tokens, tracks, files, groups, scale)

        if error < 0.01:
            print(f"  ✓ Converged")
            break

        if tokens > TARGET_TOKENS:
            hi = scale
        else:
            lo = scale

    tokens, tracks, files, groups, scale = best_result
    hours = tokens / (LMD_EVENTS / 8943)

    print(f"\n{'=' * 72}")
    print("  CALIBRATED PROFILE")
    print(f"{'=' * 72}")
    print(f"  Scale factor:        {scale:.4f}")
    print(f"  {'─' * 50}")
    print(f"  Files:               {files:>10,d}")
    print(f"  Expressive tracks:   {tracks:>10,d}")
    print(f"  Tokens:              {int(tokens):>10,d}")
    print(f"  Hours:               {hours:>10.1f}")
    print(f"  % of 50% LMD:        {tokens/TARGET_TOKENS*100:>9.1f}%")
    print(f"  % of full LMD:       {tokens/LMD_EVENTS*100:>9.1f}%")

    print(f"\n  Per-group distribution:")
    print(f"    {'Group':<23s} {'Tracks':>8s} {'%Tracks':>7s}")
    print(f"    {'-'*40}")
    for gi, tc in sorted(groups.items(), key=lambda x: -x[1]):
        pct = tc / tracks * 100
        print(f"    {gi:<23s} {tc:>8,d} {pct:>6.2f}%")
    print(f"    {'-'*40}")
    print(f"    {'TOTAL':<23s} {tracks:>8,d} {100:>6.2f}%")


if __name__ == "__main__":
    main()
