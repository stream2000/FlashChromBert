#!/bin/bash
# File: scripts/run_motif_analysis.sh
set -e

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <task_prefix> <dataset_dir> <wg_ckpt> <random_ckpt>"
    echo "Example: $0 cls data/ch_temp/promoter_finetune_data/classification/rpkm0_n_rpkm50 checkpoints/ft_promoter_cls_wg/epoch=5-val_auc=0.8770.ckpt checkpoints/ft_promoter_cls_random/epoch=6-val_auc=0.8725.ckpt"
    exit 1
fi

TASK_PREFIX=$1
DATASET_DIR=$2
WG_CKPT=$3
RANDOM_CKPT=$4

if [ "$TASK_PREFIX" == "cls" ]; then
    TASK_ARG="classification"
    SAMPLES_ARG=""
elif [ "$TASK_PREFIX" == "reg" ]; then
    TASK_ARG="regression"
    SAMPLES_ARG="--max-samples 10000"
else
    echo "Unknown task prefix: $TASK_PREFIX. Must be 'cls' or 'reg'."
    exit 1
fi

DATASET_NAME=$(basename $DATASET_DIR)
OUT_DIR_WG="logs/motif/${TASK_PREFIX}_${DATASET_NAME}_wg"
OUT_DIR_RANDOM="logs/motif/${TASK_PREFIX}_${DATASET_NAME}_random"

echo "========================================================"
echo "Starting Motif Analysis for: $DATASET_NAME (Task: $TASK_ARG)"
echo "========================================================"

source ./activate.sh

# When --max-samples is set, find_motifs must see the same subset that was
# dumped (it uses the row index from dev.tsv to index into atten.npy).
# Create a temp subset file for that case; use original otherwise.
if [ -n "$SAMPLES_ARG" ]; then
    MAX_SAMPLES=$(echo "$SAMPLES_ARG" | grep -oP '\d+')
    MOTIF_DATA_DIR=$(mktemp -d)
    head -n "$MAX_SAMPLES" "$DATASET_DIR/dev.tsv" > "$MOTIF_DATA_DIR/dev.tsv"
    # find_motifs also reads train.tsv for background sequences
    cp "$DATASET_DIR/train.tsv" "$MOTIF_DATA_DIR/train.tsv"
else
    MOTIF_DATA_DIR="$DATASET_DIR"
fi

echo "[1/4] Dumping attention for WG model..."
python scripts/utils/dump_attention.py --ckpt "$WG_CKPT" --dev-tsv "$DATASET_DIR/dev.tsv" --out-dir "$OUT_DIR_WG" --task "$TASK_ARG" --batch-size 32 $SAMPLES_ARG

echo "[2/4] Dumping attention for Random model..."
python scripts/utils/dump_attention.py --ckpt "$RANDOM_CKPT" --dev-tsv "$DATASET_DIR/dev.tsv" --out-dir "$OUT_DIR_RANDOM" --task "$TASK_ARG" --batch-size 32 $SAMPLES_ARG

echo "[3/4] Finding motifs for WG model..."
python -m flashchrombert.legacy.find_motifs --data_dir "$MOTIF_DATA_DIR" --predict_dir "$OUT_DIR_WG" --window_size 12 --min_len 5 --pval_cutoff 0.005 --min_n_motif 2 --align_all_ties --save_file_dir "$OUT_DIR_WG/result"

echo "[4/4] Finding motifs for Random model..."
python -m flashchrombert.legacy.find_motifs --data_dir "$MOTIF_DATA_DIR" --predict_dir "$OUT_DIR_RANDOM" --window_size 12 --min_len 5 --pval_cutoff 0.005 --min_n_motif 2 --align_all_ties --save_file_dir "$OUT_DIR_RANDOM/result"

[ -n "$SAMPLES_ARG" ] && rm -rf "$MOTIF_DATA_DIR"
echo "Done! Results saved in $OUT_DIR_WG/result/ and $OUT_DIR_RANDOM/result/"
