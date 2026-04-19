# FlashChromBert — Experiment Guide

Four experiments are documented here, ordered from upstream to downstream:

1. [Promoter Pre-training](#1-promoter-pre-training)
2. [Promoter Fine-tuning — Classification & Regression](#2-promoter-fine-tuning)
3. [Whole-Genome Pre-training](#3-whole-genome-pre-training)
4. [Whole-Genome Motif Extraction](#4-whole-genome-motif-extraction)

All commands assume you are at the repo root with the environment activated:

```bash
cd /home/fqijun/python/FlashChromBert
source ./activate.sh
```

---

## 1. Promoter Pre-training

**Goal**: Train a 12-layer BERT backbone on 4-mer chromatin-state sequences from promoter regions, reproducing the original ChromBERT pre-training target (val MLM loss ≈ 1.24).

### Config

```
configs/archive/ch_promoter_tuned.yaml
```

Key parameters:

| Field | Value | Note |
|---|---|---|
| tokenizer | `kmer_cstate`, k=4, states=15 | vocab 50,630 |
| masking | `kmer_mask_list`, mlm_prob=0.0375 | ×4 expansion → 15% effective |
| data | `data/ch_temp/promoter_pretrain_data/{train,val}_split.txt` | file-mode, ~2 GB RAM |
| model | 384 hidden / 12 layers / 12 heads / 1536 FFN / seq 512 | matches legacy |
| schedule | 10 k steps, 1 k warmup, lr 3e-4 | |
| hardware | 4 GPU DDP, bf16-mixed, grad_accum=2 | |

### Run

```bash
fcbert-pretrain --config configs/archive/ch_promoter_tuned.yaml
```

Checkpoints are saved to `checkpoints/ch_promoter_tuned/` (top-3 by val loss).

### Reference checkpoint

```
checkpoints/archive/ch_promoter_tuned/epoch=0-val_loss=0.257.ckpt
```

This is the backbone used by the promoter fine-tuning configs below.

---

## 2. Promoter Fine-tuning

Both tasks share the same backbone. Before running, verify that `pretrained_ckpt` in the config points to the actual file:

```
checkpoints/archive/ch_promoter_tuned/epoch=0-val_loss=0.257.ckpt
```

The configs currently reference `checkpoints/ch_promoter_tuned/...` (no `archive/` prefix) — update the path if needed.

---

### 2a. Classification

**Task**: Binary classification — not-expressed (RPKM = 0) vs. expressed (RPKM > 50).  
**Metric**: AUC (monitor on validation set).

**Config**: `configs/base/ft_promoter_cls.yaml`

Key parameters:

| Field | Value |
|---|---|
| data | `classification/rpkm0_n_rpkm50/{train,dev}.tsv` |
| batch_size | 128 |
| lr | 2e-4, warmup_ratio=0.1 |
| max_epochs | 10, 4 GPU DDP |
| classifier_dropout | 0.1 |

**Run**:

```bash
fcbert-finetune --config configs/base/ft_promoter_cls.yaml
```

**Reference checkpoint** (best val AUC = 0.877):

```
checkpoints/ft/ft_cls/production/ft_promoter_cls_wg/epoch=5-val_auc=0.8770.ckpt
```

*(This checkpoint was fine-tuned from the whole-genome backbone; the promoter-backbone equivalent is in `checkpoints/ft/ft_cls/archive_base/`.)*

---

### 2b. Regression

**Task**: Predict log-RPKM from promoter chromatin state sequence.  
**Metric**: Pearson correlation.

**Config**: `configs/base/ft_promoter_reg.yaml`

Key parameters:

| Field | Value |
|---|---|
| data | `regression/{train,dev}.tsv` (no header) |
| batch_size | 64 |
| lr | 5e-5, warmup_ratio=0.2 |
| max_epochs | 10, 4 GPU DDP |
| classifier_dropout | 0.2 |

**Run**:

```bash
fcbert-finetune --config configs/base/ft_promoter_reg.yaml
```

**Reference checkpoints** (best val Pearson = 0.740, epoch 4):

```
checkpoints/ft/ft_reg/production/ft_promoter_reg/epoch=4-val_pearson=0.7396.ckpt
```

---

## 3. Whole-Genome Pre-training

**Goal**: Pre-train on project-constructed whole-genome corpus (ROADMAP 15-state × all chromosomes, 3.1 GB / 3,041 lines). Uses streaming I/O to avoid loading 12 GB into RAM.

### Config

```
configs/base/ch_whole_genome_tuned.yaml
```

Key parameters:

| Field | Value | Note |
|---|---|---|
| tokenizer | `kmer_cstate`, k=4, states=15 | same vocab as promoter |
| masking | `kmer_mask_list`, mlm_prob=0.0375 | |
| data | `data/ch_temp/genome_pretrain_data/pretrain_genome_all.txt` | stream-mode |
| model | same 384/12/12/1536 architecture | |
| schedule | 10 k steps, 1 k warmup, lr 3e-4 | |
| hardware | devices [1,2,3] (3 GPU DDP), bf16-mixed, grad_accum=2 | |

### Run

```bash
fcbert-pretrain --config configs/base/ch_whole_genome_tuned.yaml
```

Checkpoints are saved to `checkpoints/ch_whole_genome_tuned_v2/`.

### Reference checkpoints

```
checkpoints/production/ch_whole_genome_tuned_v2/epoch=0-val_loss=0.000.ckpt
checkpoints/production/ch_whole_genome_tuned_v2/epoch=1-val_loss=0.000.ckpt
```

`val_loss=0.000` is a placeholder — streaming mode has no held-out validation split; training loss is the convergence signal.

---

## 4. Whole-Genome Motif Extraction

**Goal**: Identify chromatin-state motifs from attention weights of a model fine-tuned on top of the whole-genome backbone. The pipeline follows the original ChromBERT motif discovery approach verbatim.

### Step 0 — Fine-tune from the WG backbone

Config: `configs/base/ft_promoter_cls_wg.yaml`  
This is identical to `ft_promoter_cls.yaml` except it loads the whole-genome pre-trained checkpoint:

```
pretrained_ckpt: checkpoints/production/ch_whole_genome_tuned_v2/epoch=1-val_loss=0.000.ckpt
```

Update the path if necessary, then:

```bash
fcbert-finetune --config configs/base/ft_promoter_cls_wg.yaml
```

Reference checkpoint (val AUC = 0.877):

```
checkpoints/ft/ft_cls/production/ft_promoter_cls_wg/epoch=5-val_auc=0.8770.ckpt
```

### Step 1 — Dump last-layer attention

```bash
python scripts/utils/dump_attention.py \
    --ckpt checkpoints/ft/ft_cls/production/ft_promoter_cls_wg/epoch=5-val_auc=0.8770.ckpt \
    --dev-tsv data/ch_temp/promoter_finetune_data/classification/rpkm0_n_rpkm50/dev.tsv \
    --out-dir logs/motif/cls_rpkm0_n_rpkm50_wg \
    --task classification \
    --batch-size 32
```

Outputs `atten.npy` (L2-normalised per-base scores) and `pred_results.npy` to the specified directory.

### Step 2 — Find motifs

```bash
python -m flashchrombert.legacy.find_motifs \
    --data_dir data/ch_temp/promoter_finetune_data/classification/rpkm0_n_rpkm50 \
    --predict_dir logs/motif/cls_rpkm0_n_rpkm50_wg \
    --window_size 12 \
    --min_len 5 \
    --pval_cutoff 0.005 \
    --min_n_motif 2 \
    --align_all_ties \
    --save_file_dir logs/motif/cls_rpkm0_n_rpkm50_wg/result
```

### One-shot script (WG vs random comparison)

`scripts/motif/run_motif_analysis.sh` wraps steps 1–2 for both the WG model and a randomly-initialised baseline in a single call:

```bash
bash scripts/motif/run_motif_analysis.sh \
    cls \
    data/ch_temp/promoter_finetune_data/classification/rpkm0_n_rpkm50 \
    checkpoints/ft/ft_cls/production/ft_promoter_cls_wg/epoch=5-val_auc=0.8770.ckpt \
    checkpoints/ft/ft_cls/ablation/ft_promoter_cls_random/epoch=6-val_auc=0.8725.ckpt
```

Results land in `logs/motif/cls_rpkm0_n_rpkm50_wg/result/` and `logs/motif/cls_rpkm0_n_rpkm50_random/result/`.

### Batch analysis across multiple fine-tuned models

The legacy `find_motifs` pipeline can be run across **multiple expression-threshold splits** using `scripts/motif/batch_run_motifs.sh`. This covers four classification datasets, each fine-tuned with a different positive/negative RPKM cutoff, and compares the WG-pretrained model against a randomly-initialised baseline for each:

| Split | WG AUC | Random AUC | Description |
|---|---|---|---|
| `not_n_rpkm0` | 0.757 | 0.754 | not-expressed vs any expression |
| `not_n_rpkm50` | 0.923 | 0.922 | not-expressed vs highly expressed |
| `rpkm0_n_rpkm20` | 0.853 | 0.849 | low-expressed vs moderately expressed |
| `rpkm10_n_rpkm50` | 0.624 | 0.626 | hard boundary split |

```bash
bash scripts/motif/batch_run_motifs.sh
```

Each split produces two result directories under `logs/motif/cls_<split>_{wg,random}/result/`.

> **Note — regression motif analysis**: `scripts/motif/batch_run_reg_motifs.sh` exists but the underlying `find_motifs` script (ported verbatim from ChromBERT) only supports binary-labelled data — it splits samples by `label == 0` vs `label == 1`. Regression labels are continuous RPKM values, so no positive/negative split is possible. Use one of the classification splits above as a proxy for expression-based motif discovery.

---

## Checkpoint map

| Experiment | Config | Saved checkpoint |
|---|---|---|
| Promoter pre-train | `configs/archive/ch_promoter_tuned.yaml` | `checkpoints/archive/ch_promoter_tuned/epoch=0-val_loss=0.257.ckpt` |
| Promoter cls fine-tune | `configs/base/ft_promoter_cls.yaml` | `checkpoints/ft/ft_cls/archive_base/epoch=2-val_auc=0.8713.ckpt` |
| Promoter reg fine-tune | `configs/base/ft_promoter_reg.yaml` | `checkpoints/ft/ft_reg/production/ft_promoter_reg/epoch=4-val_pearson=0.7396.ckpt` |
| Whole-genome pre-train | `configs/base/ch_whole_genome_tuned.yaml` | `checkpoints/production/ch_whole_genome_tuned_v2/epoch=1-val_loss=0.000.ckpt` |
| WG cls fine-tune (motif) | `configs/base/ft_promoter_cls_wg.yaml` | `checkpoints/ft/ft_cls/production/ft_promoter_cls_wg/epoch=5-val_auc=0.8770.ckpt` |
