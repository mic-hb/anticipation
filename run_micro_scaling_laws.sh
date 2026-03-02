#!/bin/bash
#SBATCH -p genai-thickstun-highpri --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36GB
#SBATCH -t 100:00:00
#SBATCH -J micro_scaling_laws_amt
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e

echo "Job ID is: $SLURM_JOB_ID"

#https://github.com/karpathy/nanochat/blob/master/runs/scaling_laws.sh
FLOPS_BUDGETS=(
    1e16
    2.15e16
    4.68e16
    1e17
)
DEPTHS=(6 8 10 12 14 16)


# --- set up conda and activate it ---
# assuming conda binary lives here
CONDA_ACTIVATE_PATH="/share/apps/software/anaconda3/etc/profile.d/conda.sh"
if source "$CONDA_ACTIVATE_PATH" 2>/dev/null; then
  cd /home/mf867/anticipation_isolated/anticipation
  conda activate ./env
  echo "activated environment."
else
  echo "conda startup script not found."
fi

export OMP_NUM_THREADS=1

for flops in "${FLOPS_BUDGETS[@]}"; do
    echo "=============================================="
    echo "Compute budget: $flops FLOPs"
    echo "=============================================="

    for d in "${DEPTHS[@]}"; do
        echo "Training d=$d at $flops FLOPs..."

        TAG="scaling_${flops}_d${d}"

        # Train the model with fixed flops budget
        # The script will auto-calculate num_iterations to hit target_flops
        # CORE eval happens once at the end (999999 ensures only final step)
        PYTHONPATH=. torchrun --standalone --nproc_per_node=1 train/v2/training.py \
            --depth=$d \
            --flops=$flops \
            --output_dir "output/slurm_logs/$SLURM_JOB_ID/$TAG" \
            --gpus_per_node 1 \
            --data_dir data/tokenized_datasets/lmd_full/b82a7a2750e3c5836ffb9bf564720cd8 \
            --train_batch_size 128 \
            --eval_batch_size 128 \
            --steps_per_eval 1000 \
            --save_midi_output_after_step 100000000 \
            --steps_per_checkpoint 100000000 \
            --window_pattern "SSSL" \
            --gradient_accumulation_steps 1 \
            --pos_emb "rope" \
            --use_wandb \
            --wandb_tag "scaling"
    done
done
