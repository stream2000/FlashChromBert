# FlashChromBert

FlashChromBert is an optimized re-implementation of [ChromBERT](https://github.com/caocao0525/ChromBERT), adapted for improved engineering flexibility and training efficiency. 

This version is built upon the foundational research and core implementation of the original ChromBERT project from our group. Our goal is to provide a more modular scaffold that leverages modern PyTorch features to support large-scale genomic pre-training and faster downstream experiments.

## Core Optimizations

Based on the original design, we have introduced several engineering-focused updates:

- **Framework Integration**: Adapted the original training scripts into a structured **PyTorch Lightning** module to simplify multi-GPU training and experiment tracking.
- **Efficiency Updates**: 
  - **FlashAttention-2**: Replaced the standard attention mechanism with `torch.nn.functional.scaled_dot_product_attention` to speed up training on modern GPUs (Ampere/Ada/Hopper).
  - **BF16 Support**: Native support for mixed-precision training (BF16) to reduce memory overhead.
  - **Streaming Data**: Added `StreamingMLMDataset` to handle large-scale whole-genome datasets (e.g., IHEC 18-state data) without significant RAM requirements.
- **Consistency**:
  - Maintained the original **15-state (Roadmap)** and **18-state (IHEC)** tokenization logic.
  - Included verbatim copies of original utility modules (`css_utility.py`, etc.) in the `legacy/` directory to ensure motif analysis results remain consistent with the original paper.

## Installation

```bash
# Basic setup
pip install -e .

# For full motif analysis capabilities:
pip install -e ".[legacy]"
```

## Quick Start

### 1. Training (MLM)
```bash
# Run a baseline pre-training session
fcbert-pretrain --config configs/base/ch_promoter_tuned.yaml
```

### 2. Fine-tuning
```bash
# Fine-tune on promoter classification tasks
fcbert-finetune --config configs/base/ft_promoter_cls.yaml
```

### 3. Analysis
We provide scripts to bridge the new model checkpoints with the original analysis pipeline:
```bash
# Dump attention scores for motif identification
python scripts/utils/dump_attention.py --ckpt <path_to_ckpt> --out-dir logs/motif/output --task classification

# Run original motif finding logic
python -m flashchrombert.legacy.find_motifs --data_dir <data_path> --predict_dir logs/motif/output --save_file_dir logs/motif/output/result
```

## Acknowledgments

This project is a technical extension of the following research:
> **ChromBERT**: Learning the language of chromatin for genomic discovery.
> [Original Repository (by @caocao0525)](https://github.com/caocao0525/ChromBERT)

We thank the original authors for their pioneering work and for providing the codebase that this project is built upon.

## License

MIT License (following the original project's conventions).
