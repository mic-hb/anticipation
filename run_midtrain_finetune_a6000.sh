#!/bin/bash
#SBATCH -p thickstun --gres=gpu:nvidia_rtx_6000_ada_generation:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100GB
#SBATCH -t 100:00:00
#SBATCH -J train_ar_local_midi_v2
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e


# --- set up conda and activate it ---
# assuming conda binary lives here
CONDA_ACTIVATE_PATH="/share/apps/software/anaconda3/etc/profile.d/conda.sh"
if source "$CONDA_ACTIVATE_PATH" 2>/dev/null; then
  cd /home/ss3576/anticipation
  conda activate ./env
  echo "activated environment."
else
  echo "conda startup script not found."
fi


#export TORCH_SHOW_CPP_STACKTRACES=1
nvidia-smi


PYTHONPATH=. torchrun --standalone --nproc_per_node=2 train/v2/training.py \
    --output_dir output/checkpoints \
    --checkpoint_path output/checkpoints/pretraining-5m-transcripts-feb22/step-5000 \
    --data_dir data/tokenized_datasets/lmd_full/b82a7a2750e3c5836ffb9bf564720cd8 \
    --hidden_dim 128 \
    --num_heads 4 \
    --num_layers 4 \
    --gpus_per_node 2 \
    --train_batch_size 128 \
    --gradient_accumulation_steps 4 \
    --num_train_steps 5000 \
    --steps_per_eval 500 \
    --steps_per_checkpoint 500 \
    --bf16 \
    --learning_rate 1e-4 \
    --use_wandb