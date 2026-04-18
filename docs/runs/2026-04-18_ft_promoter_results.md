# Promoter fine-tuning: FlashChromBert vs ChromBERT paper

Fine-tunes the tuned FlashChromBert promoter pretrain checkpoint
(`checkpoints/ch_promoter_tuned/epoch=0-val_loss=0.257.ckpt`, hidden=384,
12 layers, 4-mer 15-state, 512-token window) on the two downstream tasks
from the ChromBERT paper's promoter section.

## Classification — not_expressed vs expressed (RPKM > 0)

### Harder Split: `not_n_rpkm0` (Any expression vs None)
Config: `configs/ft_promoter_cls.yaml`  ·  train 20 000 / dev 1 000
(from `data/ch_temp/promoter_finetune_data/classification/not_n_rpkm0`).
Hyperparameters: `lr=2e-5`, `bs=32`, `warmup=0.1`, `epochs=5`.

Best ckpt: `checkpoints/ft_promoter_cls/epoch=3-val_auc=0.7603.ckpt`.
Legacy Stride 1 peak: 0.76. **Parity achieved.**

### Paper Split: `rpkm0_n_rpkm50` (High expression vs None)
Config: `configs/ft_promoter_cls.yaml` updated to this split.
Hyperparameters (aligned with paper): `lr=2e-4`, `bs=32`, `warmup=0.1`, `epochs=10` on 4 × GPUs (DDP).

| epoch | val_loss | val_acc | val_f1 | val_auc |
| ----: | -------: | ------: | -----: | ------: |
|     2 |    0.413 |   0.814 |  0.817 | **0.871** |

Best ckpt: `checkpoints/ft_promoter_cls/epoch=2-val_auc=0.8713.ckpt`.
Target: AUC ≈ 0.87 (legacy Stride 1 ceiling). **P1 successfully met.**

## Regression — log-RPKM

Config: `configs/ft_promoter_reg.yaml`  ·  train 80 000 / dev 8 000
(sampled from `data/ch_temp/promoter_finetune_data/regression`).
Hyperparameters: `lr=2e-5`, `bs=32`, `warmup=0.2`, `dropout=0.2`,
`epochs=3`, bf16-mixed.

| epoch | val_loss / mse | val_pearson | val_spearman |
| ----: | -------------: | ----------: | -----------: |
|     0 |          0.971 |       0.728 |        0.747 |
|     1 |          0.948 |       0.734 |        0.753 |
|     2 |          0.945 |     **0.735** |        0.754 |

Best ckpt: `checkpoints/ft_promoter_reg/epoch=2-val_pearson=0.7351.ckpt`.

Legacy authors' regression run at Stride 1
(`data/ch_temp/promoter_finetune_result/regression/eval_results.txt`,
col 2 = val_pearson) peaks at 0.748. We are ≈0.01 below legacy Stride 1,
consistent with training on 80k of 761k available windows for 3 epochs
instead of the full set for 10 epochs.

Paper reference: peak Pearson r = 0.791 with TSS context of 100 kb
upstream + 90 kb downstream, **Stride 2**
(`chrombert-latest.txt:721-724`).

## Why FlashChromBert is below the paper headline numbers

Three effects stack. The first two are bookkeeping; only the third is a
real capability gap.

1. **Different classification split.** Paper's 0.94 AUC is on
   *highly-expressed vs non-expressed* (`rpkm0_n_rpkm50`). We ran
   `not_n_rpkm0` (any-expression vs none), which is strictly harder —
   legacy Stride 1 caps at 0.76 on it. Apples-to-apples, Stride-1 paper
   baseline on `rpkm0_n_rpkm50` is ≈ 0.87, not 0.94.

2. **Under-trained classifier.** Paper finetune uses
   `lr=2e-4`, 10 epochs for classification
   (`chrombert-latest.txt:699-700`); we used `lr=2e-5`, 5 epochs (copied
   from the regression recipe). Regression epoch budget (3 vs 10) and
   sample budget (80k vs 761k) are also short.

3. **Stride-1 context ceiling.** Even with the right split and right
   hparams, Stride 1 ≈ 6 kb around TSS. Paper's 0.94 / 0.791 require
   Stride 2/3 over 190–290 kb. Reaching those numbers needs
   regenerating finetune TSVs from raw ROADMAP 15-state tracks with
   `stride > 1` and (ideally) re-pretraining on the wider context.

What is **not** the cause: backbone size (we match legacy hidden=384 /
12 layers / 12 heads / ffn=1536) or pretrain recipe (mlm_prob=0.025,
10k steps, bs=50, all legacy-aligned — see
`project_pretrain_target_spec` memory).

## Evaluation plan to align with paper numbers

Ordered by cost, stop when the bottleneck is localized:

- **P0 — pipeline sanity.** Done. Our `not_n_rpkm0` AUC 0.760 equals
  legacy Stride 1's 0.76 → finetune code and backbone load are clean.
- **P1 — correct hparams + correct split.** Rerun classification on
  `rpkm0_n_rpkm50` with `lr=2e-4`, `epochs=10`, `warmup=0.1`. Target:
  AUC ≈ 0.87 (legacy Stride 1 ceiling on this split). If met, our
  pretrain is fully aligned with paper at Stride 1.
- **P2 — regression full budget.** `epochs=10`, drop
  `max_train_samples` (use full 761k). Target: Pearson ≈ 0.748 (legacy
  Stride 1 ceiling).
- **P3 — Stride 2/3 data regeneration.** Re-run legacy
  `reference/ChromBERT/examples/css_utility.py` with `stride ∈ {2,3}`
  against ROADMAP 15-state tracks + TSS + RPKM; finetune the existing
  backbone first, and only re-pretrain if finetune-only gains stall.
  Target: AUC ≈ 0.94 / Pearson ≈ 0.791.

## How to reproduce

```bash
./activate.sh
fcbert-finetune --config configs/ft_promoter_cls.yaml
fcbert-finetune --config configs/ft_promoter_reg.yaml
```

Metrics land in `docs/runs/2026-04-18_ft_promoter_{cls,reg}.json`; raw
step-level logs live under `lightning_logs/ft_promoter_{cls,reg}/`.
