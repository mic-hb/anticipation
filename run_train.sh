#!/bin/bash
#SBATCH -p genai-thickstun-highpri --gres=gpu:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH -t 120:00:00
#SBATCH -J train_anticipation_2xb200
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e
# sbatch run_train.sh

# --- set up conda and activate it ---
# assuming conda binary lives here
CONDA_ACTIVATE_PATH="/share/apps/software/anaconda3/etc/profile.d/conda.sh"
if source "$CONDA_ACTIVATE_PATH" 2>/dev/null; then
  cd /home/mf867/anticipation/
  conda activate ./env
  echo "activated environment."
else
  echo "conda startup script not found."
fi

# 'cuda.h is missing'
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export CPATH="/usr/local/cuda-12.8/targets/x86_64-linux/include:${CPATH:-}"

# 'no space left on device'
export TMPDIR=/share/thickstun/mf867/tmp
export TEMP=$TMPDIR
export TMP=$TMPDIR
mkdir -p "$TMPDIR"

NUM_GPUS=2
PYTHONPATH=. torchrun --standalone --nproc_per_node=$NUM_GPUS train/v2/training.py \
    --output_dir "output/slurm_logs/${SLURM_JOB_ID}/checkpoints" \
    --data_dir data/tokenized_datasets/lakh_baseline/b0d0dbce322fc3318387b6cc12cf096a \
    --gpus_per_node $NUM_GPUS \
    --eval_batch_size 128 \
    --train_batch_size 128 \
    --gradient_accumulation_steps 4 \
    --steps_per_eval 10000 \
    --steps_per_checkpoint 100 \
    --save_midi_output_after_step 1000000 \
    --num_events_to_generate_for_midi_inference 80 \
    --warmup_percent 0.01 \
    --num_layers 8 \
    --hidden_dim 768 \
    --intermediate_dim 3072 \
    --num_heads 8 \
    --no_weight_tie \
    --window_pattern "L" \
    --pos_emb "rope" \
    --learning_rate 1e-03 \
    --no_gradient_checkpointing \
    --num_train_steps 100000 \
    --mlp_style "Llama" \
    --use_value_embeds \
    --use_wandb
