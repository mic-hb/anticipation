set -e

PYTHONPATH=. torchrun --standalone --nproc_per_node=1 train/v2/training.py \
    --output_dir output/checkpoints \
    --data_dir data/tokenized_datasets/lmd_full/b82a7a2750e3c5836ffb9bf564720cd8 \
    --gpus_per_node 1 \
    --train_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --steps_per_eval 100
