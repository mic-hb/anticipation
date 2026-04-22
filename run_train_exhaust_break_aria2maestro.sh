#!/bin/bash
#SBATCH -p genai-thickstun-highpri --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH -t 140:00:00
#SBATCH -J train_exhaust_break_aria2maestro
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e
# sbatch run_train_exhaust_break_aria2maestro.sh
CHOICE="aria2maestro"

# ------------------------------------------------------------
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

# in seconds, default is 90
export WANDB_INIT_TIMEOUT=600
export USE_FA4=False
# ----
NUM_GPUS=1
STEPS_PER_VAL_REPORT=10000
BS=128
ACCUM=4
HEAD_DIM=64
ASPECT_RATIO=64

source "./run_constants.sh"

# K is used to artificially restrict number of sequences per epoch in ds2
# SEQ_MILESTONES must start with 0 (baseline) and should not exceed total sequences in ds1
case "$CHOICE" in
  aria2maestro)
    dataset1_path=$ARIA_TRAIN
    n_ds1=$ARIA_TRAIN_TOTAL_SEQ
    dataset2_path=$MAESTRO_TRAIN
    val_dataset_path=$MAESTRO_VALID
    K=("${MAESTRO_TRAIN_SUBSETS[@]}")
    SEQ_MILESTONES=("${ARIA_TRAIN_SEQ_MILESTONES[@]}")
    ;;
  aria2adl)
    dataset1_path=$ARIA_TRAIN
    n_ds1=$ARIA_TRAIN_TOTAL_SEQ
    dataset2_path=$ADL_TRAIN
    val_dataset_path=$ADL_VALID
    K=("${ADL_TRAIN_SUBSETS[@]}")
    SEQ_MILESTONES=("${ARIA_TRAIN_SEQ_MILESTONES[@]}")
    ;;
  transcripts2lakh)
    dataset1_path=$TRANSCRIPTS_TRAIN
    n_ds1=$TRANSCRIPTS_TRAIN_TOTAL_SEQ
    dataset2_path=$LAKH_TRAIN
    val_dataset_path=$LAKH_VALID
    K=("${LAKH_TRAIN_SUBSETS[@]}")
    SEQ_MILESTONES=("${TRANSCRIPTS_TRAIN_SEQ_MILESTONES[@]}")
    ;;
  *)
    echo "Unknown choice: $choice" >&2
    exit 1
    ;;
esac

declare -p NUM_LAYERS
declare -p SEQ_MILESTONES
declare -p K

echo "CONFIG CHOICE: "
echo "$CHOICE"

# same for all configs
COMMON_ARGS=(
    --dataset1_subset_seed 1234 \
    --dataset2_subset_seed 5678 \
    --ds1_num_epochs 1 \
    --dataset1_path "$dataset1_path" \
    --n_ds1 "$n_ds1" \
    --dataset2_path "$dataset2_path" \
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
    --wandb_project "${CHOICE}" \
    --use_wandb
)
# no "/" suffix
OUTPUT_DIR_PARENT="output/experiments/${CHOICE}"
for curr_layers in "${NUM_LAYERS[@]}"; do
    combo="${curr_layers}"
    output_dir="${OUTPUT_DIR_PARENT}/${combo}"

    done_file_parent="${output_dir}/DONE"

    if [[ -f "$done_file_parent" ]]; then
        echo "Skipping completed run (parent): $done_file_parent"
    else
        # run training for phase 1
        if PYTHONPATH=. python train/v2/training_exhaust.py \
            "${COMMON_ARGS[@]}" \
            --seq-milestones "${SEQ_MILESTONES[@]}" \
            --output_dir $output_dir \
            --num_layers $curr_layers
        then
            touch "$done_file_parent"
        else
            echo "Run failed: $output_dir" >&2
        fi
    fi

    # from each checkpoint, resume at phase 2, taking a subset of dataset 2
    # defined by k
    for curr_k in "${K[@]}"; do
        for milestone in "${SEQ_MILESTONES[@]}"; do
            CKPT_DIR="${output_dir}/phase1_seq-${milestone}"

            done_file="${CKPT_DIR}/DONE"
            if [[ -f "$done_file" ]]; then
                echo "Skipping completed run: $CKPT_DIR"
                continue
            fi

            CKPT_PATH="${CKPT_DIR}/trainer.ckpt"
            if [[ ! -f "${CKPT_PATH}" ]]; then
                echo "Skipping ${milestone}: checkpoint not found at ${CKPT_PATH}"
                continue
            fi

            if PYTHONPATH=. python train/v2/training_exhaust.py \
                "${COMMON_ARGS[@]}" \
                --output_dir $CKPT_DIR \
                --start_phase_2_from "${CKPT_DIR}" \
                --k_ds2 $curr_k \
                --num_layers $curr_layers
            then
                touch "$done_file"
            else
                echo "Run failed: $CKPT_DIR" >&2
            fi
        done
    done
done
