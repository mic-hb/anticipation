#!/bin/bash
#SBATCH -p thickstun --gres=gpu:nvidia_rtx_6000_ada_generation:4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=36GB
#SBATCH -t 120:00:00
#SBATCH -J train_ar_local_midi_v2
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e

echo "Job ID is: $SLURM_JOB_ID"

# --- set up conda and activate it ---
# assuming conda binary lives here
CONDA_ACTIVATE_PATH="/share/apps/software/anaconda3/etc/profile.d/conda.sh"
if source "$CONDA_ACTIVATE_PATH" 2>/dev/null; then
  cd /home/mf867/anticipation_isolated/anticipation
  conda activate ./env
  echo "activated environment."
else
  echo "conda startup script not found."
fi

# we run out of space if we just write to tmp
export CUSTOM_TMP_DIR=/scratch/$USER
mkdir -p "$CUSTOM_TMP_DIR"


# ----------------- non-preamble ---------------

# this is the place where our tokenized data is, relative to repo root
dataset_location="data/tokenized_datasets/lmd_full/b82a7a2750e3c5836ffb9bf564720cd8"
#
## move this to our scratch directory, which is faster than NFS
## note that the scratch dir depends on the node that you have obtained for the job
## basically it might not contain what you expect
#src="$dataset_location"
#dst="$CUSTOM_TMP_DIR/$dataset_location"
#
#parent="$(dirname "$dst")"
#mkdir -p "$parent"
#
## do nothing if it's already there
#if [[ ! -d "$dst" ]]; then
#    cp -R "$src" "$dst"
#fi
#
#echo "Using data_dir: "
#echo "$dst"


# NB: doesn't actually save checkpoints!!!
PYTHONPATH=. torchrun --standalone --nproc_per_node=4 train/v2/training.py \
    --output_dir "output/slurm_logs/$SLURM_JOB_ID/checkpoints" \
    --data_dir "$dataset_location" \
    --num_layers 12 \
    --gpus_per_node 4 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_heads 12 \
    --hidden_dim 768 \
    --gradient_accumulation_steps 2 \
    --steps_per_eval 1000 \
    --save_midi_output_after_step 1000000 \
    --num_events_to_generate_for_midi_inference 80 \
    --steps_per_checkpoint 10000000 \
    --use_wandb
