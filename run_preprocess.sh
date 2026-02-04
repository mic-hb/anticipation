#!/bin/bash
#SBATCH -p thickstun --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64GB
#SBATCH -t 10:00:00
#SBATCH -J midi_proeprocess
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out

# --- set up conda and activate it ---
# assuming conda binary lives here
CONDA_ACTIVATE_PATH="/share/apps/software/anaconda3/etc/profile.d/conda.sh"
if source "$CONDA_ACTIVATE_PATH" 2>/dev/null; then
  # hard coded for now
  cd /home/mf867/anticipation
  conda activate ./env
  echo "activated environment."
else
  echo "conda startup script not found."
  # cd to the location of this file on disk
  cd "$(dirname -- "$0")"
fi

set -e

PYTHONPATH=. python train/midi-preprocess.py /home/mf867/anticipation/data/lmd_full --vocab local-midi
