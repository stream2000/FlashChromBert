#!/bin/bash
set -e

WG_CKPT="checkpoints/ft_promoter_reg/epoch=6-val_pearson=0.7391.ckpt"
RANDOM_CKPT="checkpoints/ft_reg_ablation_random/epoch=3-val_pearson=0.7353.ckpt"
DATASET_DIR="data/ch_temp/promoter_finetune_data/regression"

echo "Running batch motif analysis for regression..."
./scripts/run_motif_analysis.sh reg \
    "$DATASET_DIR" \
    "$WG_CKPT" \
    "$RANDOM_CKPT"
