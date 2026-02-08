#!/bin/bash
#SBATCH -p thickstun --gres=gpu:4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36GB
#SBATCH -t 48:00:00
#SBATCH -J train_ar_local_midi_small
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

export PYTHONPATH=.
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 train_script.py --data_dir /home/mf867/anticipation/data/tokenized_new_2_4_26 --output_dir /home/mf867/anticipation/output/checkpoints --gpus_per_node=4 --train_batch_size=64 --eval_batch_size=16
