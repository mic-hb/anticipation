set -e

#PYTHONPATH=.  torchrun --standalone --nproc_per_node=4 train/v2/training.py \
#    --output_dir output/checkpoints/test_checkpoints \
#    --data_dir data/tokenized_datasets/lmd_full/b82a7a2750e3c5836ffb9bf564720cd8 \
#    --num_layers 12 \
#    --gpus_per_node 4 \
#    --train_batch_size 256 \
#    --eval_batch_size 256 \
#    --num_heads 12 \
#    --hidden_dim 768 \
#    --gradient_accumulation_steps 2 \
#    --steps_per_eval 1000 \
#    --save_midi_output_after_step 10000 \
#    --num_events_to_generate_for_midi_inference 80 \
#    --steps_per_checkpoint 100000 \
#    --use_wandb

# Lakh alone has 1,790,841,856 tokens for AR only
# Largest FLOP we can handle is 1e17
# FLOPs:
# 1e16
# 2.15e16
# 4.68e16
# 1e17

# DEPTHS:
# 2, 4, 6, 8, 10, 12, 14
PYTHONPATH=.  torchrun --standalone --nproc_per_node=1 train/v2/training.py \
    --output_dir output/checkpoints/test_checkpoints \
    --data_dir data/tokenized_datasets/lmd_full/52b9a7aa2d5d895d6e7d25740021e560 \
    --gpus_per_node 1 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --num_layers 12 \
    --gradient_accumulation_steps 1 \
    --save_midi_output_after_step 1000000 \
    --steps_per_eval 10000 \
    --steps_per_checkpoint 50000 \
    --num_train_steps 2000000 \
    --learning_rate 1e-03 \
    --no_weight_tie \
    --num_events_to_generate_for_midi_inference 80 \
    --window_pattern "SSSL" \
    --pos_emb "rope" \
    --use_wandb \
    --wandb_project "dgx_testing" \
    --no_cuda_graphs
	
