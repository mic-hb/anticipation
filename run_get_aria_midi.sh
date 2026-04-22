#!/bin/bash
set -e
# sh run_get_aria_midi.sh

# cd to the location of this file on disk, expected to be at repo root
cd "$(dirname -- "$0")"
DATA_DIR=data

mkdir -p $DATA_DIR

# download the 'pruned' split of aria-midi, to ./data/
# see here: https://github.com/loubbrad/aria-midi
wget -P $DATA_DIR https://huggingface.co/datasets/loubb/aria-midi/resolve/main/aria-midi-v1-pruned-ext.tar.gz?download=true -O ./$DATA_DIR/aria-midi-v1-pruned-ext.tar.gz

cd $DATA_DIR


# extract files to ./data/aria-midi-v1-pruned-ext/...
# is about 12.6 GB uncompressed
tar -xvf aria-midi-v1-pruned-ext.tar.gz

# delete the archive
rm aria-midi-v1-pruned-ext.tar.gz
