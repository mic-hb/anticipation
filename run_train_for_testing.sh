set -e

#PYTHONPATH=. torchrun --standalone --nproc_per_node=1 train/v2/training.py \
#    --output_dir output/checkpoints/test_checkpoints \
#    --data_dir data/tokenized_datasets/lmd_full/b82a7a2750e3c5836ffb9bf564720cd8 \
#    --gpus_per_node 1 \
#    --train_batch_size 128 \
#    --gradient_accumulation_steps 1 \
#    --steps_per_eval 100 \
#    --use_wandb

PYTHONPATH=.  torchrun --standalone --nproc_per_node=4 train/v2/training.py \
    --output_dir output/checkpoints/test_checkpoints \
    --data_dir data/tokenized_datasets/lmd_full/b82a7a2750e3c5836ffb9bf564720cd8 \
    --num_layers 12 \
    --gpus_per_node 4 \
    --train_batch_size 256 \
    --eval_batch_size 256 \
    --num_heads 12 \
    --hidden_dim 768 \
    --gradient_accumulation_steps 2 \
    --steps_per_eval 1000 \
    --save_midi_output_after_step 10000 \
    --num_events_to_generate_for_midi_inference 80 \
    --steps_per_checkpoint 100000 \
    --use_wandb

#
#PYTHONPATH=.  python train/v2/training.py \
#    --output_dir output/checkpoints/test_checkpoints \
#    --data_dir data/tokenized_datasets/lmd_full/b82a7a2750e3c5836ffb9bf564720cd8 \
#    --num_layers 1 \
#    --gpus_per_node 1 \
#    --train_batch_size 256 \
#    --eval_batch_size 256 \
#    --num_heads 12 \
#    --hidden_dim 768 \
#    --gradient_accumulation_steps 2 \
#    --steps_per_eval 1000 \
#    --save_midi_output_after_step 10000 \
#    --num_events_to_generate_for_midi_inference 80 \
#    --steps_per_checkpoint 100000 \
#    --no_ddp