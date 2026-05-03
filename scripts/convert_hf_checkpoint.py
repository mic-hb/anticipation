"""
How to use this script:

Load a Hugging Face (HF) checkpoint and convert it to a PyTorch checkpoint.

Example usage:
    python convert_hf_checkpoint.py --input_dir experiments/outputs/exp-e1/merged/standalone --output_dir experiments/outputs/exp-e1/merged/step-5000/hf

Arguments:
    --input_dir       Path to input directory containing the HF checkpoint
    --output_dir      Path to output directory for the PyTorch checkpoint
"""

from argparse import ArgumentParser
from pathlib import Path
from transformers import AutoModelForCausalLM

def parse_args():
    parser = ArgumentParser(description="Load a HF checkpoint and convert it to a PyTorch checkpoint.")
    parser.add_argument("--input_dir", required=True, help="Path to input directory containing the HF checkpoint")
    parser.add_argument("--output_dir", required=True, help="Path to output directory for the PyTorch checkpoint")
    return parser.parse_args()

def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(input_dir)
    model.save_pretrained(output_dir, safe_serialization=False)
    print(f"Converted HF checkpoint to: {output_dir}")


if __name__ == "__main__":
    main()