#!/bin/bash
set -e

cd "$(dirname -- "$0")"

CONDA_ACTIVATE_PATH="/share/apps/software/anaconda3/etc/profile.d/conda.sh"
if source "$CONDA_ACTIVATE_PATH" 2>/dev/null; then
  # hard coded for now
  cd /home/mf867/anticipation
  conda activate ./env
  echo "activated environment."
else
  echo "conda startup script not found."
fi


export PYTHONPATH=.

python scripts/eval-loss.py --filename /home/mf867/anticipation/data/tokenized_new_2_4_26/test_consolidated.txt --model /home/mf867/anticipation/output/checkpoints/step-100000/ --output ./step_100_000 --verbose --type trip
