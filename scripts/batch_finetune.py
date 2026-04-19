import argparse
import subprocess
import yaml
import os
from pathlib import Path

# Batch Fine-tuning Script for FlashChromBert
# Consolidates full batch runs, specific split runs, and random-init ablations.

DATA_DIR = Path("data/ch_temp/promoter_finetune_data/classification")
BASE_CONFIG = "configs/base/ft_promoter_cls.yaml"

def run_finetune(split_name, random_init=False, tag="batch"):
    print(f"\n>>> [{tag}] Starting Split: {split_name} (RandomInit={random_init})")
    
    if not (DATA_DIR / split_name).exists():
        print(f"!!! Error: Split directory {DATA_DIR / split_name} not found.")
        return

    # Use existing config if it's already a full path, else relative to base
    config_path = BASE_CONFIG
    if not os.path.exists(config_path):
        # Fallback to older location if needed
        config_path = "configs/ft_promoter_cls.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if random_init:
        config['pretrained_ckpt'] = None
    
    config['data']['train_file'] = str(DATA_DIR / split_name / "train.tsv")
    config['data']['val_file'] = str(DATA_DIR / split_name / "dev.tsv")
    
    suffix = f"{tag}_{split_name}"
    if random_init:
        suffix += "_random"
        
    config['trainer']['log_name'] = f"ft_cls_{suffix}"
    config['trainer']['ckpt_dir'] = f"checkpoints/ft_cls_{suffix}"
    config['trainer']['report_file'] = f"logs/batch/{suffix}_report.json"
    
    # Hyperparameter logic based on split name (following paper logic)
    if split_name.startswith("not_n_rpkm"):
        config['optimizer']['learning_rate'] = 8e-5  # Base 2e-5 * 4 (BS scaling)
        config['trainer']['max_epochs'] = 5
    else:
        config['optimizer']['learning_rate'] = 8e-4  # Base 2e-4 * 4 (BS scaling)
        config['trainer']['max_epochs'] = 10

    tmp_config = f"configs/tmp_{suffix}.yaml"
    with open(tmp_config, 'w') as f:
        yaml.dump(config, f)
    
    cmd = f"fcbert-finetune --config {tmp_config}"
    try:
        # Use executable='/bin/bash' to ensure shell features if needed
        subprocess.run(cmd, shell=True, check=True)
        print(f">>> [{tag}] Completed: {split_name}")
    except subprocess.CalledProcessError as e:
        print(f">>> [{tag}] Error: {split_name} - {e}")
    finally:
        if os.path.exists(tmp_config):
            os.remove(tmp_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch fine-tuning for ChromBERT classification tasks.")
    parser.add_argument("--splits", nargs="+", help="Specific splits to run. If empty, runs all in DATA_DIR.")
    parser.add_argument("--random", action="store_true", help="Use random initialization (ablation).")
    parser.add_argument("--tag", default="batch", help="Tag for logs and checkpoints.")
    args = parser.parse_args()

    os.makedirs("logs/batch", exist_ok=True)
    
    if args.splits:
        splits = args.splits
    else:
        exclude = ["summary.txt"]
        if DATA_DIR.exists():
            splits = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in exclude])
        else:
            print(f"!!! Error: DATA_DIR {DATA_DIR} not found.")
            splits = []

    for split in splits:
        run_finetune(split, random_init=args.random, tag=args.tag)
