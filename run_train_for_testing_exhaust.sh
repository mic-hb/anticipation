#!/bin/bash
set -e
# ./run_train_for_testing_exhaust.sh

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
NUM_GPUS=1
STEPS_PER_VAL_REPORT=10000
BS=128
ACCUM=4
HEAD_DIM=64
ASPECT_RATIO=64
NUM_LAYERS=(1 2 4 8 12)

source "./run_constants.sh"
DS_SUBSETS=("${LAKH_TRAIN_SUBSETS[@]}")
declare -p DS_SUBSETS
# make this extremely large to ensure we overfit
TOTAL_EPOCHS_FOR_OVERFIT=250
COMMON_ARGS=(
    --ds1_num_epochs $TOTAL_EPOCHS_FOR_OVERFIT \
    --dataset1_subset_seed 1234 \
    --dataset2_subset_seed 5678 \
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
    --wandb_project "basic_exhaust_debugging"
)
for curr_layers in "${NUM_LAYERS[@]}"; do
    for curr_n in "${DS_SUBSETS[@]}"; do
        combo="${curr_layers}/${curr_n}"
        echo "Training: ${combo} (layer/dataset size)"
        output_dir="output/checkpoints/basic_exhaust/${combo}"

        # run training
        # this is just phase 1, "normal" training on source dataset
        # where we restrict how many samples from it the model can
        # use to examine overfit behavior
        PYTHONPATH=. python train/v2/training_exhaust.py \
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
