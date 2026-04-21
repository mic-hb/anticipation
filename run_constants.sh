#!/bin/bash
set -e

get_total_sequences() {
    local npy_path="$1"
    local dir
    dir="$(dirname "$npy_path")"
    local filename
    filename="$(basename "$npy_path")"

    # split name (train / valid / test)
    # might not always have all 3
    local split="${filename%.npy}"
    local stats_path="${dir}/stats_${split}.json"
    if [[ ! -f "$stats_path" ]]; then
        echo "Error: stats file not found at $stats_path" >&2
        return 1
    fi

    # these are produced by the tokenization code
    val=$(jq -r '.num_sequences // empty' "$stats_path")

    if [[ -z "$val" ]]; then
        echo "Error: num_sequences not found in $stats_path" >&2
        return 1
    fi

    # return
    echo "$val"
}

generate_power_of_two_multiples() {
    local N="$1"
    local base=4096
    local val=$base

    # If N < base, return empty array
    if (( N < base )); then
        return 0
    fi

    local arr=()

    while (( val <= N )); do
        arr+=("$val")
        val=$(( val * 2 ))
    done

    # Print as space-separated values for capture
    echo "${arr[@]}"
}

# training conf
NUM_LAYERS=(2 4 8 12)

# PATHS
ARIA_TRAIN=data/tokenized_datasets/aria-midi-v1-pruned-ext/bcd5c5f66c3d0a7488711967dbd139a5/train.npy

MAESTRO_TRAIN=data/tokenized_datasets/maestro-v3.0.0/8ce2585154b0460649077706cc1013b7/train.npy
MAESTRO_VALID=data/tokenized_datasets/maestro-v3.0.0/8ce2585154b0460649077706cc1013b7/valid.npy
MAESTRO_TEST=data/tokenized_datasets/maestro-v3.0.0/8ce2585154b0460649077706cc1013b7/test.npy

ADL_TRAIN=data/tokenized_datasets/adl-piano-midi/b7e4bd861b3257f1a71cf626b2d3d98f/train.npy
ADL_VALID=data/tokenized_datasets/adl-piano-midi/b7e4bd861b3257f1a71cf626b2d3d98f/valid.npy
ADL_TEST=data/tokenized_datasets/adl-piano-midi/b7e4bd861b3257f1a71cf626b2d3d98f/test.npy

TRANSCRIPTS_TRAIN=data/tokenized_datasets/transcripts/bcd5c5f66c3d0a7488711967dbd139a5/train.npy

LAKH_TRAIN=data/tokenized_datasets/lmd_full/6cf47b1a6deebef4a6fa030e8163d5be/train.npy
LAKH_VALID=data/tokenized_datasets/lmd_full/6cf47b1a6deebef4a6fa030e8163d5be/valid.npy
LAKH_TEST=data/tokenized_datasets/lmd_full/6cf47b1a6deebef4a6fa030e8163d5be/test.npy


# --- milestones and subsets ---
# DS1s
# -- ARIA
ARIA_TRAIN_TOTAL_SEQ=$(get_total_sequences $ARIA_TRAIN)
read -r -a ARIA_TRAIN_SEQ_MILESTONES <<< "$(
    generate_power_of_two_multiples "$ARIA_TRAIN_TOTAL_SEQ"
)"
ARIA_TRAIN_SEQ_MILESTONES=(0 "${ARIA_TRAIN_SEQ_MILESTONES[@]}") # prepend 0 for baseline
declare -p ARIA_TRAIN_SEQ_MILESTONES

# -- TRANSCRIPTS
TRANSCRIPTS_TRAIN_TOTAL_SEQ=$(get_total_sequences $TRANSCRIPTS_TRAIN)
read -r -a TRANSCRIPTS_TRAIN_SEQ_MILESTONES <<< "$(
    generate_power_of_two_multiples "$TRANSCRIPTS_TRAIN_TOTAL_SEQ"
)"
TRANSCRIPTS_TRAIN_SEQ_MILESTONES=(0 "${TRANSCRIPTS_TRAIN_SEQ_MILESTONES[@]}") # prepend 0 for baseline
declare -p TRANSCRIPTS_TRAIN_SEQ_MILESTONES

# DS2s
# --- LAKH
LAKH_TRAIN_TOTAL_SEQ=$(get_total_sequences $LAKH_TRAIN)
read -r -a LAKH_TRAIN_SUBSETS <<< "$(
    generate_power_of_two_multiples "$LAKH_TRAIN_TOTAL_SEQ"
)"
LAKH_TRAIN_SUBSETS=("${LAKH_TRAIN_SUBSETS[@]}" "$LAKH_TRAIN_TOTAL_SEQ") # include full dataset
declare -p LAKH_TRAIN_SUBSETS

# --- MAESTRO
MAESTRO_TRAIN_TOTAL_SEQ=$(get_total_sequences $MAESTRO_TRAIN)
read -r -a MAESTRO_TRAIN_SUBSETS <<< "$(
    generate_power_of_two_multiples "$MAESTRO_TRAIN_TOTAL_SEQ"
)"
MAESTRO_TRAIN_SUBSETS=("${MAESTRO_TRAIN_SUBSETS[@]}" "$MAESTRO_TRAIN_TOTAL_SEQ") # include full dataset
declare -p MAESTRO_TRAIN_SUBSETS


ADL_TRAIN_TOTAL_SEQ=$(get_total_sequences $ADL_TRAIN)
read -r -a ADL_TRAIN_SUBSETS <<< "$(
    generate_power_of_two_multiples "$ADL_TRAIN_TOTAL_SEQ"
)"
ADL_TRAIN_SUBSETS=("${ADL_TRAIN_SUBSETS[@]}" "$ADL_TRAIN_TOTAL_SEQ") # include full dataset
declare -p ADL_TRAIN_SUBSETS
declare -p NUM_LAYERS

# echo "--- ARIA (DS 1) ---"
# echo $ARIA_TRAIN_TOTAL_SEQ
# echo "${ARIA_TRAIN_SEQ_MILESTONES[@]}"

# echo "--- MEASTRO (DS 2) ---"
# echo $MAESTRO_TRAIN_TOTAL_SEQ
# echo "${MAESTRO_TRAIN_SUBSETS[@]}"

# echo "--- ADL Piano (DS 2) ---"
# echo $ADL_TRAIN_TOTAL_SEQ
# echo "${ADL_TRAIN_SUBSETS[@]}"

# echo "--- Transcripts (DS 1) ---"
# echo $TRANSCRIPTS_TRAIN_TOTAL_SEQ
# echo "${TRANSCRIPTS_TRAIN_SEQ_MILESTONES[@]}"

# echo "--- Lakh (DS 2) ---"
# echo $LAKH_TRAIN_TOTAL_SEQ
# echo "${LAKH_TRAIN_SUBSETS[@]}"
