#!/bin/bash
#SBATCH -p genai-thickstun-highpri --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH -t 50:00:00
#SBATCH -J train_exhaust_aria_2_maestro
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e
# sbatch run_train_exhaust_aria_2_maestro.sh

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

# ----
NUM_GPUS=1
export USE_FA4=False

# in seconds, default is 90
export WANDB_INIT_TIMEOUT=600

SEQ_MILESTONES=(
  # 0 is baseline, train only on original dataset
  0
  2560
  5120
  10240
  20480
  40960
  81920
  163840
  327680
)

K=(
    # for aria, this is everything
    16623
)
NUM_LAYERS=(
    2
    4
    6
    8
    10
    12
    #16
)
#TOTAL_SEQ_IN_DS_1=4453603
TOTAL_SEQ_IN_DS_1=409600
STEPS_PER_VAL_REPORT=64
BS=64
ACCUM=4
HEAD_DIM=64
ASPECT_RATIO=64

# total seq in aria train: 4453603
ARIA_TRAIN=data/tokenized_datasets/aria-midi-v1-pruned-ext/b82a7a2750e3c5836ffb9bf564720cd8/train.npy

# total seq in maestro train: 16623
MAESTRO_TRAIN=data/tokenized_datasets/maestro-v3.0.0/7f1aadd4f9603af995abc3428289f7ec/train.npy
MAESTRO_VALID=data/tokenized_datasets/maestro-v3.0.0/7f1aadd4f9603af995abc3428289f7ec/valid.npy
MAESTRO_TEST=data/tokenized_datasets/maestro-v3.0.0/7f1aadd4f9603af995abc3428289f7ec/test.npy

COMMON_ARGS=(
    --dataset1_path $ARIA_TRAIN \
    --n_ds1 $TOTAL_SEQ_IN_DS_1 \
    --dataset2_path $MAESTRO_TRAIN \
    --val_dataset_path $MAESTRO_VALID \
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
    --wandb_project "amt_exhaust_aria_maestro" \
    --use_wandb
)
# no "/" suffix
OUTPUT_DIR_PARENT="output/slurm_logs/${SLURM_JOB_ID}"

for curr_layers in "${NUM_LAYERS[@]}"; do
    for curr_k in "${K[@]}"; do
        combo="${curr_layers}_${curr_k}"
        output_dir="${OUTPUT_DIR_PARENT}/${combo}"

        # run training for phase 1
        PYTHONPATH=. python train/v2/training_exhaust.py \
            "${COMMON_ARGS[@]}" \
            --seq-milestones "${SEQ_MILESTONES[@]}" \
            --k_ds2 $curr_k \
            --output_dir $output_dir \
            --num_layers $curr_layers

        # from each checkpoint, resume at phase 2
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
