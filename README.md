# FlashChromBert

Minimal modern BERT scaffold: PyTorch Lightning + BF16 + FlashAttention-2 (via `F.scaled_dot_product_attention`).

Designed to start with plain text pretraining, with clean interfaces so DNA k-mer / chromatin state tokenizers can be dropped in later without touching the model code.

## Install

```bash
mise exec python@3.11 -- python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick start

```bash
# Train a tiny MLM on the included demo text
fcbert-pretrain --config configs/tiny_text.yaml
```

## Structure

```
src/flashchrombert/
├── model/            # ~400 lines, pure PyTorch
│   ├── config.py     # BertConfig dataclass
│   ├── attention.py  # MultiHeadAttention (SDPA + eager for viz)
│   ├── embeddings.py
│   ├── encoder.py
│   ├── heads.py      # MLMHead
│   └── bert.py       # BertModel, BertForMaskedLM
├── data/
│   ├── tokenizer.py  # Tokenizer base + CharTokenizer
│   ├── dataset.py    # MLMDataset with pluggable masking
│   └── datamodule.py # LightningDataModule
├── lightning/
│   └── mlm.py        # LitBertMLM
└── cli/
    └── pretrain.py   # YAML config → Trainer.fit
```

## How to extend for chromatin state

Chromatin-state support is wired: `KmerCStateTokenizer` + `KmerMaskListMaskingStrategy`
in `src/flashchrombert/data/`, with the legacy preprocessing (`css_utility.py`)
copied verbatim under `src/flashchrombert/legacy/`. See
[`docs/LEGACY_MIGRATION.md`](docs/LEGACY_MIGRATION.md) for the migration contract
(copy-verbatim, never edit; adapters go in sibling modules).

Quick runs:
```bash
fcbert-pretrain --config configs/tiny_css.yaml      # synthetic 15-state, FA2 + bf16
fcbert-pretrain --config configs/kmer_legacy.yaml   # legacy k-mer vocab + MASK_LIST
```

## Design notes

- BF16 mixed precision by default (optimal for Ada / Hopper Tensor Cores)
- FlashAttention-2 via `F.scaled_dot_product_attention` (PyTorch 2.2+, no extra packages)
- `return_attn=True` flag on attention for motif visualization (uses eager path)
- Pre-LN transformer (training stability > BERT-original Post-LN)
