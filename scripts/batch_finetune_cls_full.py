import subprocess
import yaml
import os
import json
from pathlib import Path

# 扫描 classification 目录下的所有子文件夹作为任务
DATA_DIR = Path("data/ch_temp/promoter_finetune_data/classification")
# 排除已经跑过的和汇总文件
EXCLUDE = ["summary.txt"]
SPLITS = [d.name for d in DATA_DIR.iterdir() if d.is_dir() and d.name not in EXCLUDE]

BASE_CONFIG = "configs/ft_promoter_cls.yaml"

def run_finetune(split_name):
    print(f"\n>>> [Full Batch] Starting Split: {split_name}")
    
    with open(BASE_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    
    config['data']['train_file'] = str(DATA_DIR / split_name / "train.tsv")
    config['data']['val_file'] = str(DATA_DIR / split_name / "dev.tsv")
    config['trainer']['log_name'] = f"ft_cls_full_{split_name}"
    config['trainer']['ckpt_dir'] = f"checkpoints/ft_cls_full_{split_name}"
    config['trainer']['report_file'] = f"logs/batch_full/{split_name}_report.json"
    
    # 统一使用加速后的超参 (BS=512 总和)
    # 根据划分类型决定基础 LR
    if split_name.startswith("not_n_rpkm"):
        config['optimizer']['learning_rate'] = 8e-5  # 2e-5 * 4
        config['trainer']['max_epochs'] = 5
    else:
        config['optimizer']['learning_rate'] = 8e-4  # 2e-4 * 4
        config['trainer']['max_epochs'] = 10

    tmp_config = f"configs/tmp_full_{split_name}.yaml"
    with open(tmp_config, 'w') as f:
        yaml.dump(config, f)
    
    cmd = f"fcbert-finetune --config {tmp_config}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f">>> [Full Batch] Completed: {split_name}")
    except subprocess.CalledProcessError as e:
        print(f">>> [Full Batch] Error: {split_name} - {e}")
    finally:
        if os.path.exists(tmp_config):
            os.remove(tmp_config)

if __name__ == "__main__":
    os.makedirs("logs/batch_full", exist_ok=True)
    # 按照字母顺序排序，方便观察
    for split in sorted(SPLITS):
        run_finetune(split)
