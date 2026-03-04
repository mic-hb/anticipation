#!/bin/bash
#SBATCH -p thickstun
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50GB
#SBATCH -t 01:00:00
#SBATCH -J tokenize_dataset_lakh_10songs
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e

CONDA_ACTIVATE_PATH="/share/apps/software/anaconda3/etc/profile.d/conda.sh"
if source "$CONDA_ACTIVATE_PATH" 2>/dev/null; then
  cd /home/ss3576/anticipation
  conda activate ./env
  echo "activated environment."
else
  echo "conda startup script not found."
fi

export CUSTOM_TMP_DIR=/scratch/$USER
mkdir -p "$CUSTOM_TMP_DIR"

PYTHONPATH=. python train/v2/dataset_tokenize.py --dataset_type lmd_valid_rest