#!/bin/bash
#SBATCH -p thickstun --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH -t 100:00:00
#SBATCH -J train_ar_local_midi_v2
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

nvidia-smi

PYTHONPATH=. torchrun --standalone --nproc_per_node=1 train/v2/training2.py \
    --output_dir output/checkpoints/5m_mixed_transcripts_10song_lakh \
    --data_dir data/tokenized_datasets/transcripts/87451b329323d36a658ac64ed9a8bb81 \
    --data_dir_b data/tokenized_datasets/lmd_full/b82a7a2750e3c5836ffb9bf564720cd8 \
    --mode mixed \
    --hidden_dim 128 \
    --num_heads 4 \
    --num_layers 4 \
    --gpus_per_node 1 \
    --train_batch_size 64 \
    --gradient_accumulation_steps 2 \
    --num_train_steps 24000 \
    --steps_per_eval 1000 \
    --bf16 \
    --use_wandb