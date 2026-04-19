#!/bin/bash

set -e
# ./run_train_for_testing_exhaust.sh

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

# original global batch size is 512
# set to, for 1 gpu:
# batch_size 128
# gradient accumulation steps 4

# dataset 1 is transcripts: total 8,400,449
# dataset 2 is lakh: total 1,704,709
# validation dataset is lakh validation set

# 10240
N=(
    0
    2560
    5120
    10240
    20480
    40960
    81920
    163840
    327680
    655360
    1310720
    #2560000
)

# started at 1280
K=(
    2560
)
NUM_LAYERS=(
    2
)
# always train on transcripts for 1 epoch only
DS_1_NUM_EPOCHS=1

# train for this many epochs on Lakh / target clean dataset for the baseline
BASELINE_DS_2_NUM_EPOCHS=1000
STEPS_PER_VAL_REPORT=64
BS=128
ACCUM=4

ARIA_TRAIN=data/tokenized_datasets/aria-midi-v1-pruned-ext/b82a7a2750e3c5836ffb9bf564720cd8/train.npy
MAESTRO_TRAIN=data/tokenized_datasets/maestro-v3.0.0/7f1aadd4f9603af995abc3428289f7ec/train.npy
MAESTRO_VALID=data/tokenized_datasets/maestro-v3.0.0/7f1aadd4f9603af995abc3428289f7ec/valid.npy
MAESTRO_TEST=data/tokenized_datasets/maestro-v3.0.0/7f1aadd4f9603af995abc3428289f7ec/test.npy

for curr_layers in "${NUM_LAYERS[@]}"; do
    for curr_k in "${K[@]}"; do
        for curr_n in "${N[@]}"; do
            TOTAL_SEQUENCES=$(( curr_k * BASELINE_DS_2_NUM_EPOCHS ))
            DS1_SEQUENCES=$(( curr_n * DS_1_NUM_EPOCHS ))

            # how many steps can we allot to dataset 2?
            REMAINING_SEQUENCES=$(( TOTAL_SEQUENCES - DS1_SEQUENCES ))

            if (( REMAINING_SEQUENCES <= 0 )); then
                echo "Skipping: N=$N K=$K NUM_LAYERS=$NUM_LAYERS because DS_1 already exhausts the budget"
                continue
            fi

                # Require exact divisibility so total steps match exactly
            if (( REMAINING_SEQUENCES % curr_k != 0 )); then
                echo "Skipping: N=$N K=$K NUM_LAYERS=$NUM_LAYERS because DS_2_NUM_EPOCHS would not be an integer"
                echo "  TOTAL_SEQUENCES=$TOTAL_SEQUENCES, DS1_SEQUENCES=$DS1_SEQUENCES, REMAINING_SEQUENCES=$REMAINING_SEQUENCES"
                continue
            fi

            DS_2_NUM_EPOCHS=$(( REMAINING_SEQUENCES / curr_k ))
            echo "Running: N=$curr_n K=$curr_k NUM_LAYERS=$curr_layers DS_1_NUM_EPOCHS=$DS_1_NUM_EPOCHS DS_2_NUM_EPOCHS=$DS_2_NUM_EPOCHS"

            combo="${curr_layers}_${curr_n}_${curr_k}"
            output_dir="output/checkpoints/exhaustion_testing_break/${combo}"

            # run training
            PYTHONPATH=. python train/v2/training_exhaust.py \
                --dataset1_path $ARIA_TRAIN \
                --epochs_ds1 $DS_1_NUM_EPOCHS \
                --n_ds1 $curr_n \
                --dataset2_path $MAESTRO_TRAIN \
                --epochs_ds2 $DS_2_NUM_EPOCHS \
                --k_ds2 $curr_k \
                --val_dataset_path $MAESTRO_VALID \
                --train_batch_size $BS \
                --val_batch_size $BS \
                --gradient_accumulation_steps $ACCUM \
                --output_dir $output_dir \
                --warmup_steps 20 \
                --gpus_per_node $NUM_GPUS \
                --steps_per_eval $STEPS_PER_VAL_REPORT \
                --steps_per_checkpoint 10000 \
                --overfit_margin 0.05 \
                --wandb_project "amt_exhaustion_break_aria_maestro" \
                --use_wandb \
                --do_torch_compile \
                --pos_emb rope \
                --num_layers $curr_layers
        done
    done
done
