#!/usr/bin/env python3
"""
Merge a PEFT LoRA adapter into the base AMT model.

Produces a merged, standalone model that:
  1. Can be loaded by eval-loss.py (in step-{N}/hf/ directory structure)
  2. Can be loaded by AutoModelForCausalLM.from_pretrained() directly
  3. Has NO dependency on the PEFT library at inference time

Usage (manual pipeline):
    # After training, merge the adapter from the 'final/' or any 'checkpoint-N/' dir:
    python merge_adapter.py \\
        --adapter_dir experiments/outputs/exp-e1/final \\
        --output_dir  experiments/outputs/exp-e1/merged \\
        --step 2000

    # This creates:
    #   experiments/outputs/exp-e1/merged/step-2000/hf/  ← ready for eval-loss.py
    #   experiments/outputs/exp-e1/merged/standalone/    ← ready for from_pretrained()

Options:
    --adapter_dir     Path to the saved PEFT adapter (contains adapter_config.json)
    --output_dir      Root directory for merged output
    --base_model      Base model to merge into (default: stanford-crfm/music-large-800k)
    --step            Step number used to name the step-{N}/hf/ directory (default: 0)
    --skip_eval_fmt   Skip creating the step-{N}/hf/ layout (just save standalone)
"""

import sys
import logging
import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge PEFT LoRA adapter into base AMT model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to saved PEFT adapter directory (contains adapter_config.json)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Root output directory. Merged model is written here.",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="stanford-crfm/music-large-800k",
        help="Base model to load and merge the adapter into",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=0,
        help=(
            "Step number used to name the eval-loss.py-compatible directory: "
            "step-{N}/hf/ inside --output_dir. Use the optimizer step when the "
            "adapter was saved (e.g. 2000 for checkpoint-2000)."
        ),
    )
    parser.add_argument(
        "--skip_eval_fmt",
        action="store_true",
        help=(
            "Skip creating the step-{N}/hf/ layout. Only saves to 'standalone/' "
            "inside --output_dir. Use if you only need from_pretrained() compatibility."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    adapter_path = Path(args.adapter_dir)
    output_root = Path(args.output_dir)

    if not adapter_path.exists():
        logger.error(f"Adapter directory not found: {adapter_path}")
        sys.exit(1)

    if not (adapter_path / "adapter_config.json").exists():
        logger.error(
            f"No adapter_config.json found in {adapter_path}. "
            "Is this a valid PEFT adapter directory?"
        )
        sys.exit(1)

    output_root.mkdir(parents=True, exist_ok=True)

    # Destination paths
    standalone_path = output_root / "standalone"
    eval_fmt_path = output_root / f"step-{args.step}" / "hf"

    logger.info("=" * 60)
    logger.info("AMT LoRA Merge Utility")
    logger.info("=" * 60)
    logger.info(f"  Base model:    {args.base_model}")
    logger.info(f"  Adapter:       {adapter_path}")
    logger.info(f"  Output root:   {output_root}")
    logger.info(f"  Standalone:    {standalone_path}")
    if not args.skip_eval_fmt:
        logger.info(f"  eval-loss.py:  {eval_fmt_path}")
    logger.info("=" * 60)

    # Load base model
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading base model in {dtype} on {device}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    # Load and merge the PEFT adapter
    logger.info(f"Loading PEFT adapter from {adapter_path}...")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))

    logger.info("Merging adapter weights into base model (W + BA)...")
    merged_model = peft_model.merge_and_unload()
    logger.info("Merge complete.")

    # Save standalone version (usable with from_pretrained)
    logger.info(f"Saving standalone merged model to: {standalone_path}")
    standalone_path.mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(str(standalone_path))
    logger.info("Standalone model saved.")

    # Save eval-loss.py-compatible version (step-{N}/hf/ layout)
    if not args.skip_eval_fmt:
        logger.info(f"Saving eval-loss.py compatible model to: {eval_fmt_path}")
        eval_fmt_path.mkdir(parents=True, exist_ok=True)
        merged_model.save_pretrained(str(eval_fmt_path))
        logger.info("eval-loss.py compatible layout saved.")

    logger.info("=" * 60)
    logger.info("DONE. How to use the merged model:")
    logger.info("")
    logger.info("  [eval-loss.py — BPS metric]")
    if not args.skip_eval_fmt:
        logger.info(f"    python scripts/eval-loss.py \\")
        logger.info(f"        -f data/gigamidi_s1_10pct_random_from_all/valid.txt \\")
        logger.info(f"        -m {output_root} \\")
        logger.info(f"        -o {output_root}/eval_results.csv \\")
        logger.info(f"        --bpe -s 1")
    logger.info("")
    logger.info("  [from_pretrained — custom inference]")
    logger.info(f"    model = AutoModelForCausalLM.from_pretrained('{standalone_path}')")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
