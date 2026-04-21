#!/bin/bash

set -e
# ./run_train_for_testing_exhaust_break.sh

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

# ----
NUM_GPUS=1
export USE_FA4=False

# in seconds, default is 90
export WANDB_INIT_TIMEOUT=600

CHOICE="transcripts2lakh"
NUM_LAYERS=(
    1
    2
    4
    6
    8
    10
    12
    14
)
STEPS_PER_VAL_REPORT=320
BS=128
ACCUM=4
HEAD_DIM=64
ASPECT_RATIO=64

# --- dataset constants ----
ARIA_TRAIN=data/tokenized_datasets/aria-midi-v1-pruned-ext/c823210ee9c21e83f8c5d233975ab85b/train.npy
ARIA_TRAIN_TOTAL_SEQ=4307585

MAESTRO_TRAIN=data/tokenized_datasets/maestro-v3.0.0/e361f9c323930538df6ba7762bf4dc9f/train.npy
MAESTRO_VALID=data/tokenized_datasets/maestro-v3.0.0/e361f9c323930538df6ba7762bf4dc9f/valid.npy
MAESTRO_TEST=data/tokenized_datasets/maestro-v3.0.0/e361f9c323930538df6ba7762bf4dc9f/test.npy
MAESTRO_TRAIN_TOTAL_SEQ=16765

ADL_TRAIN=data/tokenized_datasets/adl-piano-midi/c823210ee9c21e83f8c5d233975ab85b/train.npy
ADL_VALID=data/tokenized_datasets/adl-piano-midi/c823210ee9c21e83f8c5d233975ab85b/valid.npy
ADL_TEST=data/tokenized_datasets/adl-piano-midi/c823210ee9c21e83f8c5d233975ab85b/test.npy
ADL_TRAIN_TOTAL_SEQ=28800

LAKH_TRAIN=data/tokenized_datasets/lmd_full/ad9826395376a4e7c9be1eb6e07c45b6/train.npy
LAKH_VALID=data/tokenized_datasets/lmd_full/ad9826395376a4e7c9be1eb6e07c45b6/valid.npy
LAKH_TEST=data/tokenized_datasets/lmd_full/ad9826395376a4e7c9be1eb6e07c45b6/test.npy
LAKH_TRAIN_TOTAL_SEQ=1718700

TRANSCRIPTS_TRAIN=data/tokenized_datasets/transcripts/c823210ee9c21e83f8c5d233975ab85b/train.npy
TRANSCRIPTS_TRAIN_TOTAL_SEQ=8405028

# K is used to artificially restrict number of sequences per epoch in ds2
# SEQ_MILESTONES must start with 0 (baseline) and should not exceed total sequences in ds1
case "$CHOICE" in
  aria2maestro)
    dataset1_path=$ARIA_TRAIN
    n_ds1=$ARIA_TRAIN_TOTAL_SEQ
    dataset2_path=$MAESTRO_TRAIN
    val_dataset_path=$MAESTRO_VALID
    K=(4191 8382 16765)
    SEQ_MILESTONES=(0 2560 5120 10240 20480 40960 81920 163840 327680 655360 1310720 2621440)
    ;;
  aria2adl)
    dataset1_path=$ARIA_TRAIN
    n_ds1=$ARIA_TRAIN_TOTAL_SEQ
    dataset2_path=$ADL_TRAIN
    val_dataset_path=$ADL_VALID
    K=(2560 5120)
    SEQ_MILESTONES=(0 2560 5120 10240 20480 40960 81920 163840 327680 655360 1310720 2621440)
    ;;
  transcripts2lakh)
    dataset1_path=$TRANSCRIPTS_TRAIN
    n_ds1=$TRANSCRIPTS_TRAIN_TOTAL_SEQ
    dataset2_path=$LAKH_TRAIN
    val_dataset_path=$LAKH_VALID
    K=(2560 5120)
    SEQ_MILESTONES=(0 2560 5120 10240 20480 40960 81920 163840 327680 655360 1310720 2621440 5242880)
    ;;
  *)
    echo "Unknown choice: $choice" >&2
    exit 1
    ;;
esac

echo "CONFIG CHOICE: "
echo "$CHOICE"

# same for all configs
COMMON_ARGS=(
    --ds1_num_epochs 1 \
    --dataset1_subset_seed 1234 \
    --dataset1_path "$dataset1_path" \
    --n_ds1 "$n_ds1" \
    --dataset2_path "$dataset2_path" \
    --dataset2_subset_seed 4567 \
    --val_dataset_path "$val_dataset_path" \
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
    --wandb_project "final_exhaust_4_debugging" \
    --use_wandb
)
# no "/" suffix
OUTPUT_DIR_PARENT="output/checkpoints/final_exhaust_4/${CHOICE}"
for curr_layers in "${NUM_LAYERS[@]}"; do
    combo="${curr_layers}"
    output_dir="${OUTPUT_DIR_PARENT}/${combo}"

    # run training for phase 1
    PYTHONPATH=. python train/v2/training_exhaust.py \
        "${COMMON_ARGS[@]}" \
        --seq-milestones "${SEQ_MILESTONES[@]}" \
        --output_dir $output_dir \
        --num_layers $curr_layers \
        --check_vocabs

    # from each checkpoint, resume at phase 2, taking a subset of dataset 2
    # defined by k
    for curr_k in "${K[@]}"; do
        for milestone in "${SEQ_MILESTONES[@]}"; do
            CKPT_DIR="${output_dir}/phase1_seq-${milestone}"
            CKPT_PATH="${CKPT_DIR}/trainer.ckpt"
            if [[ ! -f "${CKPT_PATH}" ]]; then
                echo "Skipping ${milestone}: checkpoint not found at ${CKPT_PATH}"
                continue
            fi

            PYTHONPATH=. python train/v2/training_exhaust.py \
                "${COMMON_ARGS[@]}" \
                --output_dir $CKPT_DIR \
                --start_phase_2_from "${CKPT_DIR}" \
                --k_ds2 $curr_k
        done
    done
done
