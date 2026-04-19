#!/bin/bash
#SBATCH -p genai-thickstun-highpri --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH -t 120:00:00
#SBATCH -J train_exhaust
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e
# sbatch run_train_exhaust.sh

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
#
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
    25600
)

# started at 1280
K=(
    2560
    5120
    10240
    20480
    40960
    81920
)
NUM_LAYERS=(
    2
    4
)
DS_1_NUM_EPOCHS=1
DS_2_NUM_EPOCHS=1000
STEPS_PER_VAL_REPORT=64
BS=128
ACCUM=4

for curr_k in "${K[@]}"; do
    for curr_layers in "${NUM_LAYERS[@]}"; do
        for curr_n in "${N[@]}"; do
            combo="${curr_layers}_${curr_n}_${curr_k}_exhaust"
            echo "Training: ${combo}"

            output_dir="output/slurm_logs/${SLURM_JOB_ID}/${combo}"

            # run training
            PYTHONPATH=. torchrun --standalone --nproc_per_node=$NUM_GPUS train/v2/training_exhaust.py \
                --dataset1_path ddata/tokenized_datasets/transcripts/7f1aadd4f9603af995abc3428289f7ec/train.npy \
                --epochs_ds1 $DS_1_NUM_EPOCHS \
                --n_ds1 $curr_n \
                --dataset2_path data/tokenized_datasets/lakh_baseline/b0d0dbce322fc3318387b6cc12cf096a/train.npy \
                --epochs_ds2 $DS_2_NUM_EPOCHS \
                --k_ds2 $curr_k \
                --val_dataset_path data/tokenized_datasets/lakh_baseline/b0d0dbce322fc3318387b6cc12cf096a/valid.npy \
                --train_batch_size $BS \
                --val_batch_size $BS \
                --gradient_accumulation_steps $ACCUM \
                --output_dir $output_dir \
                --gpus_per_node $NUM_GPUS \
                --steps_per_eval $STEPS_PER_VAL_REPORT \
                --warmup_steps 20 \
                --steps_per_checkpoint 10000 \
                --wandb_project "amt_exhaustion_v2" \
                --use_wandb \
                --num_layers $curr_layers
        done
    done
done
