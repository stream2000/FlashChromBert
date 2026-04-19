# FlashChromBert

FlashChromBert is a modern, high-performance migration and optimization of [ChromBERT](https://github.com/caocao0525/ChromBERT). It is designed to be a minimal yet powerful scaffold for genomic pre-training and downstream fine-tuning, leveraging the latest deep learning features for maximum efficiency.

## Key Features & Enhancements

- **Modern Architecture**: Migrated from legacy scripts to a structured, modular design using **PyTorch Lightning**.
- **Performance Optimized**:
  - **FlashAttention-2**: Integrated via `torch.nn.functional.scaled_dot_product_attention` for significantly faster training and reduced memory footprint (requires PyTorch 2.2+).
  - **BF16 Mixed Precision**: Native support for `bf16-mixed` precision, optimized for modern GPUs (NVIDIA Ada/Hopper architectures).
  - **Pre-LN Transformer**: Switched from original Post-LN to **Pre-LN** configuration for superior training stability.
- **Scalability**:
  - **Streaming Datasets**: Implemented `StreamingMLMDataset` to handle massive datasets (e.g., 40GB+ Whole Genome data) without loading everything into RAM.
  - **DDP Ready**: Fully compatible with Distributed Data Parallel (DDP) for multi-GPU scaling.
- **Analysis Toolkit**:
  - **Integrated Motif Analysis**: Ported and adapted the original ChromBERT motif analysis pipeline (`find_motifs`, `css_utility`).
  - **Attention Visualization**: Built-in support for dumping last-layer attention for biological interpretation and motif clustering.
  - **Linear Probing**: Easy-to-use interfaces for ablation studies between pre-trained and random-init backbones.

## Installation

We recommend using [mise](https://mise.jdx.dev/) for environment management.

```bash
# Setup environment
pip install -e .

# For motif analysis and legacy compatibility, install extra dependencies:
pip install -e ".[legacy]"
```

## Quick Start

### 1. Pre-training (MLM)
Train on chromatin state sequences (15-state ROADMAP or 18-state IHEC):
```bash
# Tiny test run
fcbert-pretrain --config configs/base/ch_promoter_tuned.yaml
```

### 2. Fine-tuning
Perform downstream tasks like gene expression classification or regression:
```bash
fcbert-finetune --config configs/base/ft_promoter_cls.yaml
```

### 3. Motif Analysis
Dump attention scores and run the motif identification pipeline:
```bash
# 1. Dump attention
python scripts/utils/dump_attention.py --ckpt <path_to_ckpt> --out-dir logs/motif/my_run --task classification

# 2. Find motifs
python -m flashchrombert.legacy.find_motifs --data_dir <dataset_dir> --predict_dir logs/motif/my_run --save_file_dir logs/motif/my_run/result
```

## Project Structure

```
src/flashchrombert/
├── model/            # Pure PyTorch implementation with FlashAttention
├── data/             # Tokenizers, Datamodules, and Streaming Datasets
├── lightning/        # LightningModule wrappers for MLM and Finetune
├── cli/              # Command-line interfaces for training
└── legacy/           # Verbatim copies of ChromBERT utility modules for analysis
configs/              # YAML configurations for various tasks
scripts/              # Analysis, batch processing, and utility scripts
```

## Credits

This project is a reimplementation of the research presented in:
> **ChromBERT**: Learning the language of chromatin for genomic discovery.
> [Original Repository](https://github.com/caocao0525/ChromBERT)

## License

MIT License. See `LICENSE` for details (if applicable).
