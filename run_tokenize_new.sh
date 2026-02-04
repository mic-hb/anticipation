#!/bin/bash
#SBATCH -p thickstun
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -t 05:00:00
#SBATCH -J tokenize_new
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

#PYTHONPATH=. python train/tokenize-new.py /home/mf867/anticipation/data/lmd_full --vocab local-midi
#PYTHONPATH=. python train/tokenize-new.py /home/mf867/anticipation/data/lmd_full /home/mf867/anticipation/data/tokenized_new_2_4_26 autoregress 1024 --vocab local-midi --workers 8

cd train
export PYTHONPATH=..

# generate valid split
./tokenize.sh /home/mf867/anticipation/data/lmd_full /home/mf867/anticipation/data/tokenized_new_2_4_26 autoregress 1024 valid 1 local-midi "" 8

# generate train split
#./tokenize.sh /home/mf867/anticipation/data/lmd_full /home/mf867/anticipation/data/tokenized_new_2_4_26 autoregress 1024 train 1 local-midi "" 8

