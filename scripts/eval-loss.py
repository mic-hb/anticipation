import os
import csv
import time
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F

from transformers import AutoModelForCausalLM
from tqdm import tqdm

from anticipation.config import EVENT_SIZE, TIME_RESOLUTION
from anticipation.ops import max_time
from anticipation.vocab import SEPARATOR


def compute_dataset_hours_arrival(datafile: str, subsample: int) -> tuple[float, int, int]:
    """
    Total music duration (hours) for lines that would be evaluated at this subsample.
    Matches dataset-stats.py logic for arrival-time: skip SEPARATOR lines, use max_time(tokens[1:], seconds=False).
    Returns (hours, lines_used, total_time_bins).
    """
    total_time_bins = 0
    lines_used = 0
    with open(datafile, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % subsample != 0:
                continue
            tokens = [int(t) for t in line.split()]
            if SEPARATOR in tokens:
                continue
            if len(tokens) < 2:
                continue
            total_time_bins += max_time(tokens[1:], seconds=False)
            lines_used += 1
    hours = total_time_bins / (TIME_RESOLUTION * 3600.0) if total_time_bins else 0.0
    return float(hours), lines_used, int(total_time_bins)


def log_loss(model, datafile, subsample):
    with open(datafile, 'r') as data:
        ce = torch.empty(0)
        for i,line in tqdm(list(enumerate(data))):
            if i % subsample != 0:
                continue

            tokens = [int(token) for token in line.split()]
            tokens = torch.tensor(tokens).unsqueeze(0).cuda()
            with torch.no_grad():
                logits = model(tokens).logits[0]
                ce = torch.cat([ce, F.cross_entropy(logits[:-1],tokens[0,1:],reduction='none').cpu()])

    return ce


def main(args):
    print(f'Sub-sampling results at rate {args.subsample}')

    results = os.path.join(args.model, args.output)
    print(f'Storing results at {results}')

    checkpoints = [os.path.join(f.path, 'hf') for f in os.scandir(args.model) if
            f.is_dir() and os.path.basename(f).startswith('step-')]

    if args.all:
        print('Calculating log-loss for checkpoints:')
        for ckpt in checkpoints:
            print('  ', ckpt)
    else:
        steps = [int(ckpt.split(os.sep)[-2][5:]) for ckpt in checkpoints]
        checkpoints = [os.path.join(args.model, f'step-{max(steps)}', 'hf')]
        print('Calculating log-loss for final checkpoint:')
        print('  ', checkpoints[0])

    print('Calculating log-loss on dataset:')
    print('  ', args.filename)

    hours_for_bpe = None
    hours_meta = {}
    if args.bpe:
        if args.hours is not None:
            hours_for_bpe = float(args.hours)
            hours_meta = {
                "hours_source": "cli",
                "hours": hours_for_bpe,
            }
        elif args.interarrival:
            raise SystemExit(
                "Automatic hour computation is only implemented for arrival-time data. "
                "Pass --hours explicitly when using -i/--interarrival."
            )
        else:
            print("Scanning dataset for total duration (hours) for bpe denominator ...")
            t_scan = time.time()
            hours_for_bpe, lines_used, total_time_bins = compute_dataset_hours_arrival(
                args.filename, args.subsample
            )
            hours_meta = {
                "hours_source": "computed_from_file",
                "hours": hours_for_bpe,
                "lines_used_for_hours": lines_used,
                "total_time_bins": total_time_bins,
                "scan_seconds": round(time.time() - t_scan, 3),
            }

    print("--- eval metadata (before model load / loss) ---")
    print(f"  dataset: {args.filename}")
    print(f"  basename: {os.path.basename(args.filename)}")
    print(f"  subsample: {args.subsample}")
    print(f"  interarrival: {bool(args.interarrival)}")
    print(f"  bpe requested: {bool(args.bpe)}")
    if args.bpe:
        for k, v in hours_meta.items():
            print(f"  {k}: {v}")
    print(f"  model root: {args.model}")
    print(f"  checkpoints: {len(checkpoints)}")
    for ckpt in checkpoints:
        print(f"    - {ckpt}")
    print("--- end metadata ---")

    with open(results, 'w', newline='') as f:
        fields = ['step', 'loss']
        if args.bpe:
            fields.extend(['bpe', 'hours_used_for_bpe'])
        if not args.interarrival:
            fields.extend(['event_ppl', 'onset_ppl', 'dur_ppl', 'note_ppl'])

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for ckpt in checkpoints:
            step = int(ckpt.split(os.sep)[-2][5:])
            print(f'Loading checkpoint (step {step}):')
            print('  ', ckpt)
            t0 = time.time()
            model = AutoModelForCausalLM.from_pretrained(ckpt).cuda()
            print(f'  loaded in {time.time()-t0} seconds')

            ce = log_loss(model, args.filename, args.subsample)

            res = {}
            res['step'] = step
            res['loss'] = np.round(ce.mean().item(), 3)
            if args.bpe:
                if hours_for_bpe is None or hours_for_bpe <= 0:
                    raise SystemExit("bpe requires positive hours; use --hours or fix dataset scan.")
                denom_seconds = hours_for_bpe * 3600.0
                res['bpe'] = args.subsample * ce.mean().item() * np.log2(np.e) * (len(ce) / denom_seconds)
                res['hours_used_for_bpe'] = round(hours_for_bpe, 6)
                print(
                    f"  bpe: {res['bpe']:.6f} (hours_used_for_bpe={res['hours_used_for_bpe']}, "
                    f"len_ce={len(ce)})"
                )
            if not args.interarrival:
                res['event_ppl'] = np.round(np.exp(EVENT_SIZE*ce.mean().item()), 3)
                res['onset_ppl'] = np.round(np.exp(ce[0::3].mean().item()), 3)
                res['dur_ppl'] = np.round(np.exp(ce[1::3].mean().item()), 3)
                res['note_ppl'] = np.round(np.exp(ce[2::3].mean().item()), 3)

            writer.writerow(res)


if __name__ == '__main__':
    parser = ArgumentParser(description='evaluate log-loss for a tokenized dataset')
    parser.add_argument('-f', '--filename', help='file containing a tokenized dataset')
    parser.add_argument('-m', '--model', help='file containing a model to evaluate')
    parser.add_argument('-o', '--output', help='output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose console output')
    parser.add_argument('-a', '--all', action='store_true',
            help='calculate loss for all checkpoints')
    parser.add_argument('--bpe', action='store_true',
            help='Also compute bps-style bpe metric (uses --hours or auto-scanned hours from -f).')
    parser.add_argument('-i', '--interarrival', action='store_true',
            help='request interarrival-time enocoding (default to arrival-time encoding)')
    parser.add_argument('-s', '--subsample', type=int, default=10,
            help='dataset subsampling ratio')
    parser.add_argument(
        '--hours',
        type=float,
        default=None,
        help='Total dataset duration in hours for bpe denominator. If omitted with --bpe '
             '(arrival-time only), hours are computed from -f by summing max_time per evaluated line.',
    )

    main(parser.parse_args())
