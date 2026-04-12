#!/bin/bash

set -e
# ./run_train_for_testing_multi.sh

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

# original global batch size is 512
# set to, for 1 gpu:
# batch_size 128
# gradient accumulation steps 4

# dataset 1 is transcripts: total 8,400,449
# dataset 2 is lakh: total 1,704,709
# validation dataset is lakh validation set

N=(
    8203
    16407
    32814
    65628
    131257
    262514
)

K=(
    6659
    13318
    26636
    53272
)

subsample_for_test=100

for curr_k in "${K[@]}"; do
    for curr_n in "${N[@]}"; do
        echo "Training: n=${curr_n}, k=${curr_k}"

        combo="${curr_n}_${curr_k}"
        output_dir="output/checkpoints/midtraining_testing/${combo}"

        # run training
        PYTHONPATH=. python train/v2/training_multi.py \
            --dataset1_path data/tokenized_datasets/transcripts/7f1aadd4f9603af995abc3428289f7ec/train.npy \
            --n_ds1 $curr_n \
            --dataset2_path data/tokenized_datasets/lakh_baseline/b0d0dbce322fc3318387b6cc12cf096a/train.npy \
            --k_ds2 $curr_k \
            --val_dataset_path data/tokenized_datasets/lakh_baseline/b0d0dbce322fc3318387b6cc12cf096a/valid.npy \
            --train_batch_size 128 \
            --val_batch_size 128 \
            --gradient_accumulation_steps 4 \
            --output_dir $output_dir \
            --gpus_per_node $NUM_GPUS \
            --steps_per_eval 100000000 \
            --warmup_steps 20 \
            --steps_per_checkpoint 1000 \
            --wandb_project "midtraining_amt" \
            --num_layers 2

        # run eval
        PYTHONPATH=. python eval/v2/eval_loss.py \
            --checkpoint_dir $output_dir \
            --subsample $subsample_for_test
    done
done
