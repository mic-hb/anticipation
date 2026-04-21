#!/bin/bash
#SBATCH -p genai-thickstun-highpri --gres=gpu:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH -t 200:00:00
#SBATCH -J train_exhaust_lakh
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e
# sbatch run_train_exhaust_lakh.sh

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

export USE_FA4=False
# --------------------------

NUM_GPUS=2
STEPS_PER_VAL_REPORT=10000
BS=128
ACCUM=4
HEAD_DIM=64
ASPECT_RATIO=64

source "./run_constants.sh"
DS_SUBSETS=("${LAKH_TRAIN_SUBSETS[@]}")

# make this extremely large to ensure we overfit
TOTAL_EPOCHS_FOR_OVERFIT=250
COMMON_ARGS=(
    --dataset1_subset_seed 1234 \
    --dataset2_subset_seed 5678 \
    --ds1_num_epochs $TOTAL_EPOCHS_FOR_OVERFIT \
    --train_batch_size $BS \
    --val_batch_size $BS \
    --gradient_accumulation_steps $ACCUM \
    --warmup_steps 20 \
    --gpus_per_node $NUM_GPUS \
    --steps_per_eval $STEPS_PER_VAL_REPORT \
    --steps_per_checkpoint 10000 \
    --overfit_margin 0.05 \
    --do_torch_compile \
    --aspect_ratio $ASPECT_RATIO \
    --head_dim $HEAD_DIM \
    --pos_emb rope \
    --wandb_project "lakh_exhaust" \
    --use_wandb
)
# num layers is defined in ./run_constant.sh
for curr_layers in "${NUM_LAYERS[@]}"; do
    for curr_n in "${DS_SUBSETS[@]}"; do
        combo="${curr_layers}/${curr_n}"
        echo "Training: ${combo}"
        output_dir="output/experiments/lakh_exhaust/${combo}"

        # run training
        # this is just phase 1, "normal" training on source dataset
        # where we restrict how many samples from it the model can
        # use to examine overfit behavior
        PYTHONPATH=. torchrun --standalone --nproc_per_node=$NUM_GPUS train/v2/training_exhaust.py \
            "${COMMON_ARGS[@]}" \
            --dataset1_path $LAKH_TRAIN \
            --n_ds1 $curr_n \
            --dataset2_path $LAKH_TRAIN \
            --k_ds2 0 \
            --val_dataset_path $LAKH_VALID \
            --output_dir $output_dir \
            --num_layers $curr_layers
    done
done
