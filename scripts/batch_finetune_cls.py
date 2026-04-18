import subprocess
import yaml
import os
from pathlib import Path

# 待验证的关键划分 (Split)
# 选取原则：涵盖 困难(not_n_rpkm0), 论文基线(rpkm0_n_rpkm50), 以及中等难度(rpkm0_n_rpkm20)
SPLITS = [
    "not_n_rpkm0",
    "rpkm0_n_rpkm20",
    "rpkm0_n_rpkm50"
]

BASE_CONFIG = "configs/ft_promoter_cls.yaml"

def run_finetune(split_name):
    print(f"\n>>> Starting Fine-tuning for Split: {split_name}")
    
    # 加载基础配置
    with open(BASE_CONFIG, 'r') as f:
        config = yaml.safe_load(f)
    
    # 更新特定参数
    config['data']['train_file'] = f"data/ch_temp/promoter_finetune_data/classification/{split_name}/train.tsv"
    config['data']['val_file'] = f"data/ch_temp/promoter_finetune_data/classification/{split_name}/dev.tsv"
    config['trainer']['log_name'] = f"ft_cls_batch_{split_name}"
    config['trainer']['ckpt_dir'] = f"checkpoints/ft_cls_batch_{split_name}"
    
    # 针对不同划分调整超参 (对齐论文逻辑)
    # 基础 LR 分别为 2e-5 和 2e-4 (针对单卡 BS=32)
    # 现在 BS=128 (4x)，按线性缩放理论建议提高 LR
    # 这里我们采用保守策略：rpkm0_n_rpkm50 任务提高到 8e-4
    if split_name == "not_n_rpkm0":
        config['optimizer']['learning_rate'] = 8e-5  # 2e-5 * 4
        config['trainer']['max_epochs'] = 5
    else:
        config['optimizer']['learning_rate'] = 8e-4  # 2e-4 * 4
        config['trainer']['max_epochs'] = 10

    # 写入临时配置
    tmp_config = f"configs/tmp_ft_{split_name}.yaml"
    with open(tmp_config, 'w') as f:
        yaml.dump(config, f)
    
    # 执行命令 (4卡 DDP)
    cmd = f"fcbert-finetune --config {tmp_config}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f">>> Completed Split: {split_name}")
    except subprocess.CalledProcessError as e:
        print(f">>> Error in Split {split_name}: {e}")
    finally:
        if os.path.exists(tmp_config):
            os.remove(tmp_config)

if __name__ == "__main__":
    os.makedirs("logs/batch", exist_ok=True)
    for split in SPLITS:
        run_finetune(split)
