#!/bin/bash
#SBATCH -p genai-thickstun-highpri --gres=gpu:4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36GB
#SBATCH -t 48:00:00
#SBATCH -J train_ar_local_midi_v2
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.outset
set -e

# cd to the location of this file on disk
cd "$(dirname -- "$0")"

# --- set up conda and activate it ---
# assuming conda binary lives here
CONDA_ACTIVATE_PATH="/share/apps/software/anaconda3/etc/profile.d/conda.sh"
if source "$CONDA_ACTIVATE_PATH" 2>/dev/null; then
  conda activate ./env
  echo "activated environment."
else
  echo "conda startup script not found."
fi


#export TORCH_SHOW_CPP_STACKTRACES=1

PYTHONPATH=. torchrun --standalone --nproc_per_node=4 train/v2/training.py \
    --output_dir output/checkpoints \
    --data_dir data/tokenized_data/5dbc372cdeae4c4fb44f447e66029461 \
    --gpus_per_node 4 \
    --train_batch_size 256 \
    --gradient_accumulation_steps 1 \
    --use_wandb
