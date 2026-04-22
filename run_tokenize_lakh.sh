#!/bin/bash
#SBATCH -p thickstun
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=50GB
#SBATCH -t 01:00:00
#SBATCH -J tokenize_dataset_lakh_ar_only
#SBATCH -e output/slurm_logs/%j/stderr.err
#SBATCH -o output/slurm_logs/%j/stdout.out
set -e
# ./run_tokenize_lakh.sh


init_conda() {
  # helper: source file if it exists
  _try_source() {
    local path="$1"
    [[ -f "$path" ]] || return 1
    # shellcheck disable=SC1090
    source "$path"
  }

  # Prefer conda’s reported base, if conda is on PATH
  if command -v conda >/dev/null 2>&1; then
    local base
    base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${base:-}" ]] && _try_source "$base/etc/profile.d/conda.sh"; then
      return 0
    fi
  fi

  # Fallbacks
  _try_source "$HOME/miniconda3/etc/profile.d/conda.sh" && return 0
  _try_source "$HOME/anaconda3/etc/profile.d/conda.sh" && return 0

  echo "Error: could not locate conda or conda.sh. Install conda or add it to PATH." >&2
  return 1
}

if ! init_conda; then
  exit 1
fi

conda activate ./env

# we run out of space if we just write to tmp
CTD=/scratch/$USER
if mkdir -p "$CTD" 2>/dev/null && [ -d "$CTD" ] && [ -w "$CTD" ]; then
    export CUSTOM_TMP_DIR="$CTD"
else
    echo "Warning: directory $CTD is not usable" >&2
fi

PYTHONPATH=. python train/v2/dataset_tokenize.py --dataset_type lakh --v1_mode
