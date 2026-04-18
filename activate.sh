#!/usr/bin/env bash
# Activate the flashchrombert conda env and fix the LD_LIBRARY_PATH order
# so that torch's bundled NVIDIA libs (libnvJitLink, libcusparse, etc.) are
# loaded from site-packages instead of an older system copy.
#
# Usage:  source ./activate.sh

source /work/miniconda3/etc/profile.d/conda.sh
conda activate flashchrombert

NVIDIA_LIB_ROOT="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
if [[ -d "$NVIDIA_LIB_ROOT" ]]; then
    for lib_dir in "$NVIDIA_LIB_ROOT"/*/lib; do
        if [[ -d "$lib_dir" ]]; then
            export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH:-}"
        fi
    done
fi

echo "[flashchrombert] env active — python: $(which python)"
echo "[flashchrombert] LD_LIBRARY_PATH prefixed with torch-bundled NVIDIA libs"
