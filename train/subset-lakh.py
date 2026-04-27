import random
import shutil
from argparse import ArgumentParser
from pathlib import Path


def parse_args():
    parser = ArgumentParser(description="Create a percentage subset mirror of Lakh MIDI with split structure preserved.")
    parser.add_argument("src_dir", help="Path to source lmd_full directory")
    parser.add_argument("dst_dir", help="Path to destination subset directory")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.10,
        help="Fraction of files to keep per split directory (default: 0.10)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    return parser.parse_args()


def main():
    args = parse_args()

    src_root = Path(args.src_dir)
    dst_root = Path(args.dst_dir)

    if not src_root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_root}")
    if args.ratio <= 0 or args.ratio > 1:
        raise ValueError(f"--ratio must be in (0, 1], got {args.ratio}")

    random.seed(args.seed)
    split_names = "0123456789abcdef"

    for split in split_names:
        (dst_root / split).mkdir(parents=True, exist_ok=True)

    total_kept = 0
    total_seen = 0
    for split in split_names:
        src_split = src_root / split
        dst_split = dst_root / split
        if not src_split.exists():
            print(f"{split}: source split not found, skipping")
            continue

        files = sorted([p for p in src_split.iterdir() if p.suffix.lower() in [".mid", ".midi"]])
        if not files:
            print(f"{split}: no midi files")
            continue

        keep = max(1, int(len(files) * args.ratio))
        chosen = random.sample(files, keep)
        for src_file in chosen:
            shutil.copy2(src_file, dst_split / src_file.name)

        total_kept += keep
        total_seen += len(files)
        print(f"{split}: kept {keep}/{len(files)} ({keep / len(files):.2%})")

    print(f"Done. Kept {total_kept}/{total_seen} files ({(total_kept / total_seen) if total_seen else 0:.2%})")
    print(f"Subset root: {dst_root}")


if __name__ == "__main__":
    main()
