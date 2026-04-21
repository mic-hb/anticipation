#!/bin/bash
set -e

# cd to the location of this file on disk
cd "$(dirname -- "$0")"

# run all pytest suite
PYTHONPATH=. pytest tests/

# speciifc test
#PYTHONPATH=. pytest tests/v2/test_tokenize.py::test_tokenize_no_controls_with_force_piano
