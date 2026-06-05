#!/usr/bin/env python3
"""
Scan GigaMIDI V2.0 dataset and compute real statistics from actual .mid files.

Strategy (per user request):
  Phase 1: Walk all .mid files → exact file count + total size + per-split breakdown
  Phase 2: Parse a uniform random sample → estimate events, tracks, duration
           Extrapolate to full dataset using the exact file count.

Raw binary MIDI parser is used for speed (~500 files/s/core).
"""

import argparse
import json
import os
import random
import struct
import sys
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

import mido

TOKENS_PER_EVENT = 3.0


def read_varlen(data, offset):
    """Read MIDI variable-length value, return (value, new_offset)."""
    value = 0
    while True:
        byte = data[offset]
        offset += 1
        value = (value << 7) | (byte & 0x7F)
        if not (byte & 0x80):
            break
    return value, offset


def scan_midi_raw(filepath):
    """Parse a MIDI file using raw binary. Returns dict with note_events and tracks."""
    try:
        with open(filepath, "rb") as f:
            data = f.read()
    except Exception:
        return None
    if len(data) < 14 or data[:4] != b"MThd":
        return None

    note_events = 0
    track_count = 0
    offset = 14  # start of first track chunk

    while offset + 8 <= len(data):
        if data[offset:offset+4] != b"MTrk":
            break
        track_count += 1
        trk_len = struct.unpack_from(">I", data, offset + 4)[0]
        trk_end = offset + 8 + trk_len
        if trk_end > len(data):
            break
        pos = offset + 8

        running_status = 0
        while pos < trk_end:
            delta, pos = read_varlen(data, pos)
            if pos >= trk_end:
                break
            status = data[pos]
            if status & 0x80:
                running_status = status
                pos += 1
            else:
                status = running_status
                if status == 0:
                    break

            msg_type = status & 0xF0
            if msg_type == 0x90:
                if pos + 2 > trk_end:
                    break
                vel = data[pos + 1]
                if vel > 0:
                    note_events += 1
                pos += 2
            elif msg_type == 0x80:
                pos += 2
            elif msg_type in (0xA0, 0xB0, 0xE0):
                pos += 2
            elif msg_type == 0xC0:
                pos += 1
            elif msg_type == 0xD0:
                pos += 1
            elif status == 0xFF:
                pos += 1
                if pos >= trk_end:
                    break
                meta_len, pos = read_varlen(data, pos)
                pos += meta_len
            elif status in (0xF0, 0xF7):
                sysex_len, pos = read_varlen(data, pos)
                pos += sysex_len
            else:
                break

        offset = trk_end

    return {
        "note_events": note_events,
        "tracks": track_count,
    }


def scan_midi_mido(filepath):
    """Parse a MIDI file using mido for duration."""
    try:
        mid = mido.MidiFile(str(filepath))
        if mid.ticks_per_beat <= 0:
            return {"duration_sec": 0.0}
        try:
            dur_sec = mid.length
        except Exception:
            # Fallback: manual duration calc at 120 BPM
            max_tick = 0
            for track in mid.tracks:
                t = 0
                for msg in track:
                    t += msg.time
                if t > max_tick:
                    max_tick = t
            dur_sec = max_tick * 500000 / mid.ticks_per_beat / 1_000_000
        return {"duration_sec": dur_sec}
    except Exception:
        return None


def discover_files(root):
    """Walk the directory tree and find all .mid files.
    Returns (file_paths_list, total_size_bytes, per_split_counts).
    """
    file_paths = []
    total_size = 0
    split_counts = Counter()
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".mid") and not fn.startswith("._"):
                full = os.path.join(dirpath, fn)
                file_paths.append(full)
                total_size += os.path.getsize(full)
    file_paths.sort()
    return file_paths, total_size


def fmt(n):
    if isinstance(n, float):
        return f"{n:,.2f}"
    return f"{int(n):,}"


def main():
    parser = argparse.ArgumentParser(
        description="Scan GigaMIDI V2.0 and compute real statistics"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="/home/developer/auto-midi/lib/anticipation/data/Final_GigaMIDI_V2.0_Final",
        help="GigaMIDI dataset root directory",
    )
    parser.add_argument(
        "--sample-pct",
        type=float,
        default=5.0,
        help="%% of files to parse for events/tracks (default: 5.0)",
    )
    parser.add_argument(
        "--duration-sample",
        type=int,
        default=500,
        help="Number of files to parse with mido for duration (default: 500)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, cpu_count() - 1),
        help="Number of parallel workers for parsing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--parse-all",
        action="store_true",
        help="Parse ALL files (slow, ~1 hour). Overrides --sample-pct.",
    )
    args = parser.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists():
        print(f"Error: directory {root} does not exist")
        sys.exit(1)

    sys.stdout.reconfigure(line_buffering=True)

    print("=" * 72)
    print("  GigaMIDI V2.0 — Real Statistics (file scan mode)")
    print("=" * 72)
    print(f"  Root:   {root}")
    print()

    # ── Phase 1: Walk all files ──────────────────────────────────────
    print("Phase 1: Walking directory tree...")
    t0 = time.time()
    all_files, total_size = discover_files(root)
    n_total = len(all_files)
    t1 = time.time()
    print(f"  Found {n_total:,} .mid files ({t1-t0:.1f}s)")
    print(f"  Total size: {total_size/(1024**3):.2f} GB")

    # Per-split breakdown
    splits = [d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")]
    print(f"\n  Per-split breakdown:")
    for s in sorted(splits):
        s_files = [f for f in all_files if str(f).startswith(str(s) + os.sep)]
        s_size = sum(os.path.getsize(f) for f in s_files)
        print(f"    {s.name:<45s} {len(s_files):>10,d} files  {s_size/(1024**3):>7.2f} GB")

    # ── Phase 2: Sample parse ────────────────────────────────────────
    if args.parse_all:
        sample_files = all_files
        pct_label = "100% (ALL)"
    else:
        pct = args.sample_pct / 100.0
        n_sample = max(1, int(n_total * pct))
        random.seed(args.seed)
        sample_files = random.sample(all_files, n_sample)
        pct_label = f"{args.sample_pct}%"

    print(f"\nPhase 2: Parsing {pct_label} sample ({len(sample_files):,} files)...")
    t0 = time.time()

    sample_events_total = 0
    sample_tracks_total = 0
    sample_parsed = 0
    sample_errors = 0

    with Pool(args.workers) as pool:
        for result in pool.imap_unordered(scan_midi_raw, sample_files, chunksize=200):
            sample_parsed += 1
            if result is None:
                sample_errors += 1
                continue
            sample_events_total += result["note_events"]
            sample_tracks_total += result["tracks"]

            if sample_parsed % max(1, len(sample_files)//20) == 0:
                pct_done = sample_parsed / len(sample_files) * 100
                sys.stdout.write(f"\r  Progress: {sample_parsed:,}/{len(sample_files):,} ({pct_done:.0f}%)")
                sys.stdout.flush()

    t1 = time.time()
    print(f"\r  Parsed {sample_parsed:,} files in {t1-t0:.1f}s "
          f"({sample_parsed/(t1-t0):.0f} files/s, {sample_errors} errors)")
    print(f"  Sample events: {sample_events_total:,}   Sample tracks: {sample_tracks_total:,}")

    # ── Phase 3: Duration subsample (mido) ──────────────────────────
    dur_sample_size = min(args.duration_sample, n_total)
    random.seed(args.seed + 1)
    dur_sample_files = random.sample(all_files, dur_sample_size)
    print(f"\nPhase 3: Duration subsample ({dur_sample_size:,} files with mido)...")
    t0 = time.time()
    dur_total = 0.0
    dur_parsed = 0
    dur_errors = 0

    for f in dur_sample_files:
        result = scan_midi_mido(f)
        dur_parsed += 1
        if result is None:
            dur_errors += 1
            continue
        dur_total += result["duration_sec"]
        if dur_parsed % 100 == 0:
            sys.stdout.write(f"\r  Progress: {dur_parsed}/{dur_sample_size}")
            sys.stdout.flush()

    t1 = time.time()
    print(f"\r  Parsed {dur_parsed:,} files in {t1-t0:.1f}s ({dur_errors} errors)")
    avg_dur_sec = dur_total / max(1, dur_parsed)
    print(f"  Average duration: {avg_dur_sec:.2f} sec/file")

    # ── Compute final metrics ────────────────────────────────────────
    ratio = n_total / len(sample_files) if len(sample_files) > 0 else 1
    estimated_total_events = sample_events_total * ratio
    estimated_total_tracks = sample_tracks_total * ratio
    estimated_total_tokens = estimated_total_events * TOKENS_PER_EVENT
    estimated_total_hours = avg_dur_sec * n_total / 3600

    # For full parse, ratio = 1, so estimates are exact
    label = "ESTIMATED" if not args.parse_all else "EXACT"

    # ── Print results ────────────────────────────────────────────────
    print()
    print("=" * 72)
    print(f"  GIGAMIDI V2.0 — {label} METRICS" +
          (f"  (sample: {args.sample_pct}%)" if not args.parse_all else ""))
    print("=" * 72)
    print(f"  {'Metric':<35s} {'Value':>20s}")
    print(f"  {'-' * 57}")

    print(f"\n  -- Volume --")
    print(f"  {'Total files':<35s} {fmt(n_total):>20s}")
    print(f"  {'Total size (GB)':<35s} {fmt(total_size/(1024**3)):>20s}")
    print(f"  {'Total tracks (sequences)':<35s} {fmt(estimated_total_tracks):>20s}")
    print(f"  {'Total note_on events':<35s} {fmt(estimated_total_events):>20s}")
    print(f"  {'Total tokens (arrival-time ×3)':<35s} {fmt(estimated_total_tokens):>20s}")
    print(f"  {'Total duration (hours)':<35s} {fmt(estimated_total_hours):>20s}")

    print(f"\n  -- Averages --")
    avg_eph = estimated_total_events / estimated_total_hours if estimated_total_hours else 0
    avg_tph = estimated_total_tokens / estimated_total_hours if estimated_total_hours else 0
    print(f"  {'Avg events/hour':<35s} {fmt(avg_eph):>20s}")
    print(f"  {'Avg tokens/hour':<35s} {fmt(avg_tph):>20s}")
    print(f"  {'Avg events/file':<35s} {fmt(estimated_total_events/n_total):>20s}")
    print(f"  {'Avg tracks/file':<35s} {fmt(estimated_total_tracks/n_total):>20s}")
    print(f"  {'Avg duration/file (sec)':<35s} {fmt(avg_dur_sec):>20s}")
    print(f"  {'Avg file size (KB)':<35s} {fmt(total_size/n_total/1024):>20s}")

    print(f"\n  -- Accuracy --")
    print(f"  {'Files scanned (exact)':<35s} {fmt(n_total):>20s}")
    print(f"  {'Files parsed':<35s} {fmt(len(sample_files)):>20s}")
    print(f"  {'Parse errors (skipped)':<35s} {fmt(sample_errors):>20s}")
    print(f"  {'Duration sample size':<35s} {fmt(dur_parsed):>20s}")
    print(f"  {'Duration errors':<35s} {fmt(dur_errors):>20s}")

    # Comparison with Lakh MIDI
    print()
    print("=" * 72)
    print("  COMPARISON WITH LAKH MIDI DATASET")
    print("=" * 72)
    lmd_files = 178_165
    lmd_events = 663_555_310
    lmd_tokens = 1_990_665_930
    lmd_hours = 8943

    print(f"  {'Metric':<35s} {'Lakh MIDI':>14s} {'GigaMIDI V2':>14s} {'Ratio':>8s}")
    print(f"  {'-' * 73}")
    for name, lmd_v, gig_v in [
        ("Files", lmd_files, n_total),
        ("Events", lmd_events, estimated_total_events),
        ("Tokens (arrival)", lmd_tokens, estimated_total_tokens),
        ("Hours", lmd_hours, estimated_total_hours),
    ]:
        ratio_val = gig_v / lmd_v if lmd_v else 0
        print(f"  {name:<35s} {fmt(lmd_v):>14s} {fmt(gig_v):>14s}  {ratio_val:>6.1f}x")

    print()
    for name, lmd_v, gig_v in [
        ("Events/hour", lmd_events / lmd_hours, avg_eph),
        ("Tokens/hour", lmd_tokens / lmd_hours, avg_tph),
        ("Avg events/file", lmd_events / lmd_files, estimated_total_events / n_total),
    ]:
        print(f"  {name:<35s} {fmt(lmd_v):>14s} {fmt(gig_v):>14s}")

    if args.save:
        result_data = {
            "strategy": "full_scan" if args.parse_all else f"sample_{args.sample_pct}pct",
            "total_files": n_total,
            "files_parsed": len(sample_files),
            "parse_errors": sample_errors,
            "total_tracks": estimated_total_tracks,
            "total_events": estimated_total_events,
            "total_tokens": estimated_total_tokens,
            "total_hours": estimated_total_hours,
            "total_size_gb": total_size / (1024 ** 3),
            "avg_events_per_hour": avg_eph,
            "avg_tokens_per_hour": avg_tph,
            "avg_events_per_file": estimated_total_events / n_total,
            "avg_tracks_per_file": estimated_total_tracks / n_total,
            "avg_duration_sec_per_file": avg_dur_sec,
            "avg_file_size_kb": total_size / n_total / 1024,
            "is_exact": args.parse_all,
        }
        with open(args.save, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"\n  Results saved to: {args.save}")

    print("=" * 72)


if __name__ == "__main__":
    main()
