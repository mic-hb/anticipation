#!/bin/bash
set -e
# sh run_get_lakh.sh

# cd to the location of this file on disk, expected to be at repo root
cd "$(dirname -- "$0")"

DATA_DIR=data

mkdir -p $DATA_DIR

DL_LINK="http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
wget -P $DATA_DIR "$DL_LINK" -O $DATA_DIR/lmd_full.tar.gz

cd $DATA_DIR

# extract files to ./data/lmd_full
tar -xvf lmd_full.tar.gz

# delete the archive
rm lmd_full.tar.gz
