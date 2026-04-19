#!/bin/bash
set -e

SPLITS=(
    "not_n_rpkm0"
    "not_n_rpkm50"
    "rpkm0_n_rpkm20"
    "rpkm10_n_rpkm50"
)

declare -A WG_CKPTS
WG_CKPTS["not_n_rpkm0"]="checkpoints/ft/ft_cls/batch_runs/ft_cls_full_not_n_rpkm0/epoch=4-val_auc=0.7570.ckpt"
WG_CKPTS["not_n_rpkm50"]="checkpoints/ft/ft_cls/batch_runs/ft_cls_full_not_n_rpkm50/epoch=2-val_auc=0.9232.ckpt"
WG_CKPTS["rpkm0_n_rpkm20"]="checkpoints/ft/ft_cls/batch_runs/ft_cls_full_rpkm0_n_rpkm20/epoch=2-val_auc=0.8529.ckpt"
WG_CKPTS["rpkm10_n_rpkm50"]="checkpoints/ft/ft_cls/batch_runs/ft_cls_full_rpkm10_n_rpkm50/epoch=5-val_auc=0.6240.ckpt"

declare -A RANDOM_CKPTS
RANDOM_CKPTS["not_n_rpkm0"]="checkpoints/ft/ft_cls/ablation/ft_cls_ablation_not_n_rpkm0/epoch=4-val_auc=0.7540.ckpt"
RANDOM_CKPTS["not_n_rpkm50"]="checkpoints/ft/ft_cls/ablation/ft_cls_ablation_not_n_rpkm50/epoch=2-val_auc=0.9219.ckpt"
RANDOM_CKPTS["rpkm0_n_rpkm20"]="checkpoints/ft/ft_cls/ablation/ft_cls_ablation_rpkm0_n_rpkm20/epoch=2-val_auc=0.8485.ckpt"
RANDOM_CKPTS["rpkm10_n_rpkm50"]="checkpoints/ft/ft_cls/ablation/ft_cls_ablation_rpkm10_n_rpkm50/epoch=1-val_auc=0.6258.ckpt"

for split in "${SPLITS[@]}"; do
    echo "Running batch motif analysis for $split..."
    ./scripts/motif/run_motif_analysis.sh cls \
        "data/ch_temp/promoter_finetune_data/classification/$split" \
        "${WG_CKPTS[$split]}" \
        "${RANDOM_CKPTS[$split]}"
done
