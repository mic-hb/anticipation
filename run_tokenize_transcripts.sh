#!/bin/bash
#SBATCH -p thickstun
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128GB
#SBATCH -t 05:00:00
#SBATCH -J tokenize_dataset_transcripts
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e

# sbatch run_tokenize_transcripts.sh

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
export CUSTOM_TMP_DIR=/scratch/$USER
mkdir -p "$CUSTOM_TMP_DIR"

PYTHONPATH=. python train/v2/dataset_tokenize.py \
    --dataset_type transcripts \
    --v1_mode \
    --settings_json_name "ar_only_local_midi_no_instr_limit_settings_87451b329323d36a658ac64ed9a8bb81.json" \

# validation split is the Lakh split
cp "data/tokenized_datasets/lakh_baseline/b0d0dbce322fc3318387b6cc12cf096a/valid.npy" "data/tokenized_datasets/transcripts/7f1aadd4f9603af995abc3428289f7ec/valid.npy"
