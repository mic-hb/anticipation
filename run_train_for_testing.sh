set -e
# sh run_train_for_testing.sh

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

NUM_GPUS=1
PYTHONPATH=. torchrun --standalone --nproc_per_node=$NUM_GPUS train/v2/training.py \
    --output_dir output/checkpoints/test_checkpoints/baseline_test \
    --data_dir data/tokenized_datasets/lakh_baseline/b0d0dbce322fc3318387b6cc12cf096a \
    --gpus_per_node $NUM_GPUS \
    --train_batch_size 16 \
    --eval_batch_size 16 \
    --gradient_accumulation_steps 8 \
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
    --bf16 \
    --flops "2e20" \
    --use_value_embeds \
    --mlp_style "Llama"
