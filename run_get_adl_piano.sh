#!/bin/bash
# sh run_get_adl_piano.sh
set -e

# cd to the location of this file on disk
# assuming it is located at REPO_ROOT/setup/<FILENAME>.sh
cd "$(dirname -- "$0")"
DATA_DIR=data

mkdir -p $DATA_DIR

# from this helpful paper/repo: https://github.com/lucasnfe/adl-piano-midi/tree/master
wget -O $DATA_DIR/adl-piano-midi.zip https://github.com/lucasnfe/adl-piano-midi/raw/refs/heads/master/midi/adl-piano-midi.zip

# puts this into ./data/adl-piano-midi.zip
# there are some duplicates internally, -o says yes to overwrite them as they come up
unzip -o $DATA_DIR/adl-piano-midi.zip -d $DATA_DIR/

echo "You may delete the file ${DATA_DIR}/adl-piano-midi.zip. We did not delete it for safety."
