#!/bin/bash
# sh run_get_maestro.sh
set -e

# cd to the location of this file on disk
# assuming it is located at REPO_ROOT/setup/<FILENAME>.sh
cd "$(dirname -- "$0")"
DATA_DIR=data

mkdir -p $DATA_DIR

# download from: https://magenta.withgoogle.com/datasets/maestro#v300
wget -O $DATA_DIR/maestro-v3.0.0-midi.zip https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip

# puts this into ./data/maestro-v3.0.0
unzip $DATA_DIR/maestro-v3.0.0-midi.zip -d $DATA_DIR/

echo "You may delete the file ${DATA_DIR}/maestro-v3.0.0-midi.zip. We did not delete it for safety."
