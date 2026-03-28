set -e
# sh run_train_for_testing.sh

# PYTHONPATH=.  torchrun --standalone --nproc_per_node=$NUM_GPUS train/v2/training.py \
#     --output_dir output/checkpoints/test_checkpoints \
#     --data_dir data/tokenized_datasets/giga_midi/6fb2094dfa7c0d16278dfaa4a401e3b8 \
#     --gpus_per_node $NUM_GPUS \
#     --checkpoint_path "/home/mf867/anticipation_isolated/anticipation/output/slurm_logs/791545/checkpoints/step-40000" \
#     --train_batch_size 128 \
#     --eval_batch_size 128 \
#     --gradient_accumulation_steps 4 \
#     --steps_per_eval 1000 \
#     --steps_per_checkpoint 20000 \
#     --save_midi_output_after_step 1000000 \
#     --num_events_to_generate_for_midi_inference 80 \
#     --warmup_percent 0.01 \
#     --num_layers 8 \
#     --hidden_dim 768 \
#     --intermediate_dim 3072 \
#     --num_heads 8 \
#     --no_weight_tie \
#     --window_pattern "L" \
#     --pos_emb "rope" \
#     --learning_rate 1e-03 \
#     --flops "2e20" \
#     --mlp_style "Llama"
#--checkpoint_path "/home/mf867/anticipation_isolated/anticipation/output/slurm_logs/791545/checkpoints/step-40000" \

NUM_GPUS=2
PYTHONPATH=. torchrun --standalone --nproc_per_node=$NUM_GPUS train/v2/training.py \
    --output_dir output/checkpoints/test_checkpoints/resume_test \
    --checkpoint_path "output/checkpoints/test_checkpoints/resume_test/step-15" \
    --data_dir data/tokenized_datasets/giga_midi/6fb2094dfa7c0d16278dfaa4a401e3b8 \
    --gpus_per_node $NUM_GPUS \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --steps_per_eval 1000 \
    --steps_per_checkpoint 15 \
    --save_midi_output_after_step 1000000 \
    --num_events_to_generate_for_midi_inference 80 \
    --warmup_percent 0.01 \
    --num_layers 8 \
    --hidden_dim 768 \
    --intermediate_dim 3072 \
    --num_heads 8 \
    --no_weight_tie \
    --window_pattern "L" \
    --pos_emb "rope" \
    --learning_rate 1e-03 \
    --flops "2e20" \
    --mlp_style "Llama"
