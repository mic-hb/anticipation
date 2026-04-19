#!/bin/bash
#SBATCH -p thickstun --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH -t 120:00:00
#SBATCH -J train_exhaust_break_tiny
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e
# sbatch run_train_exhaust_break_tiny.sh

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

# transcripts
# 0%
# ~1%
# 10%
N=(
    655360
    1310720
)

# lakh
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
BS=32
ACCUM=16

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

            if (( REMAINING_SEQUENCES % curr_k != 0 )); then
                echo "Skipping: N=$N K=$K NUM_LAYERS=$NUM_LAYERS because DS_2_NUM_EPOCHS would not be an integer"
                echo "  TOTAL_SEQUENCES=$TOTAL_SEQUENCES, DS1_SEQUENCES=$DS1_SEQUENCES, REMAINING_SEQUENCES=$REMAINING_SEQUENCES"
                continue
            fi

            DS_2_NUM_EPOCHS=$(( REMAINING_SEQUENCES / curr_k ))
            echo "Running: N=$curr_n K=$curr_k NUM_LAYERS=$curr_layers DS_1_NUM_EPOCHS=$DS_1_NUM_EPOCHS DS_2_NUM_EPOCHS=$DS_2_NUM_EPOCHS"

            combo="${curr_layers}_${curr_n}_${curr_k}"
            output_dir="output/slurm_logs/${SLURM_JOB_ID}/${combo}"

            # run training
            CUDA_LAUNCH_BLOCKING=1 PYTHONPATH=. python train/v2/training_exhaust.py \
                --dataset1_path data/tokenized_datasets/transcripts/7f1aadd4f9603af995abc3428289f7ec/train.npy \
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
                --steps_per_checkpoint 1000 \
                --wandb_project "amt_exhaustion_break" \
                --use_wandb \
                --num_layers $curr_layers
        done
    done
done
