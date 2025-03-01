#!/bin/bash
# Specify the directory (or datafile) to shard
input_directory="$1"

# Specify the number of shards
k="$3"

echo "Files to be sharded ($k shards per file):"
find "$input_directory"*.train.txt -type f | sed 's/^/ /'

shard_directory="$2"
mkdir -p $shard_directory/tmp

echo "Output to: $shard_directory"

# Function to split a file into k shards
split_file() {
    file=$1
    shard_directory="$2"
    k=$3
    total_lines=$(wc -l < "$file")
    ((lines_per_shard = (total_lines + k - 1) / k))

    # Split the file into k shards
    split -d -a 3 -l "$lines_per_shard" "$file" "${shard_directory}/$(basename "$file").shard-"
}

# Function to concatenate and shuffle the k'th shards
process_shards() {
    output_file="$2/train.shard-$1.txt"

    # Concatenate the k'th shard of each original file
    shard_number=$(printf "%03d" "$1")
    cat "$2"/tmp/*.shard-${shard_number} > "${output_file}"

    # Shuffle the contents of the concatenated file
    shuf "${output_file}" -o "${output_file}"
}

export -f split_file
export -f process_shards

# Step 1: Fork n processes to split each file into k shards
find "$input_directory"*.train.txt -type f -exec bash -c 'split_file "$0" "$1" "$2"' {} "$shard_directory/tmp" "$k" \;
wait

# Step 2: Fork k processes to process each group of shards
max_parallel_jobs=10
for (( i=0; i<k; i++ ))
do
    # Start the process in the background
    (process_shards $i $shard_directory) &

    # Check if we have reached the maximum number of parallel jobs
    if (( (i + 1) % max_parallel_jobs == 0 )); then
        # Wait for all background jobs to finish before continuing
        wait
    fi
done

wait

# Clean up intermediate files
rm -r $shard_directory/tmp
