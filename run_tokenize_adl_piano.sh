#!/bin/bash
#SBATCH -p thickstun
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH -t 05:00:00
#SBATCH -J tokenize_dataset_adl_piano
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e

# sbatch run_tokenize_adl_piano.sh

# --- set up conda and activate it ---
# assuming conda binary lives here
CONDA_ACTIVATE_PATH="/share/apps/software/anaconda3/etc/profile.d/conda.sh"
if source "$CONDA_ACTIVATE_PATH" 2>/dev/null; then
  cd /home/mf867/anticipation
  conda activate ./env
  echo "activated environment."
else
  echo "conda startup script not found."
fi


# we run out of space if we just write to tmp
CTD=/scratch/$USER
if mkdir -p "$CTD" 2>/dev/null && [ -d "$CTD" ] && [ -w "$CTD" ]; then
    export CUSTOM_TMP_DIR="$CTD"
else
    echo "Warning: directory $CTD is not usable" >&2
fi

PYTHONPATH=. python train/v2/dataset_tokenize.py \
    --dataset_type adl_piano \
    --v1_mode
