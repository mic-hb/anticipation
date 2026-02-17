#!/bin/bash
set -e

# cd to the location of this file on disk
cd "$(dirname -- "$0")"

PYTHONPATH=. python train/v2/training.py --output_dir output/try --data_dir data/tokenized_data/cf4f32e5e408407088ae2dbea7258f99