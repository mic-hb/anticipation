#!/bin/bash
#SBATCH -p genai-thickstun-highpri --gres=gpu:2
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128GB
#SBATCH -t 120:00:00
#SBATCH -J train_multi
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e
# sbatch run_train_multi.sh

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

NUM_GPUS=2
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
    525028
)

K=(
    6659
    13318
    26636
    53272
)

# original eval script default is 10
subsample_for_test=20
num_layers=8

for curr_k in "${K[@]}"; do
    for curr_n in "${N[@]}"; do
        echo "Training: n=${curr_n}, k=${curr_k}"

        # set these for consistency
        combo="${curr_n}_${curr_k}"
        output_dir="output/slurm_logs/${SLURM_JOB_ID}/${combo}"

        # steps_per_eval is so large it won't happen, just collect at the end
        PYTHONPATH=. torchrun --standalone --nproc_per_node=$NUM_GPUS train/v2/training_multi.py \
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
            --steps_per_eval 10000000 \
            --warmup_steps 20 \
            --steps_per_checkpoint 1000 \
            --wandb_project "midtraining_amt" \
            --use_wandb \
            --num_layers $num_layers

        # run eval
        PYTHONPATH=. python eval/v2/eval_loss.py \
            --checkpoint_dir $output_dir \
            --subsample $subsample_for_test
    done
done
