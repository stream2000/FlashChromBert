# FlashChromBert 启动子微调操作手册

本手册对应 `docs/promoter.md` 列出的两个下游任务：

- **基因表达二分类（dnaprom）**：高/低表达判别。
- **基因表达回归（gene_expression）**：直接预测 log-RPKM。

微调复用已训好的启动子 MLM backbone（`checkpoints/ch_promoter_tuned/*.ckpt`），在其上加 CLS pooler + 线性 head。入口统一为 `fcbert-finetune --config <yaml>`。

---

## 1. 环境

同 `PRETRAIN_OPERATIONS.md`：

```bash
cd /home/fqijun/python/FlashChromBert
source ./activate.sh
```

首次使用需安装评测依赖（已在当前环境完成）：

```bash
pip install scikit-learn scipy
```

CLI 注册在 `pyproject.toml`：

```toml
[project.scripts]
fcbert-pretrain = "flashchrombert.cli.pretrain:main"
fcbert-finetune = "flashchrombert.cli.finetune:main"
```

任何修改 `src/` 后执行 `pip install -e .` 以刷新 entry point。

---

## 2. 数据布局

微调数据已通过 legacy 预处理脚本生成，存放于：

```
data/ch_temp/promoter_finetune_data/
├── classification/
│   ├── not_n_rpkm0/{train.tsv, dev.tsv}      # 默认：不表达(=0) vs 表达(>0)
│   ├── not_n_rpkm10/...                      # 其他阈值组合
│   └── rpkmX_n_rpkmY/...
└── regression/
    ├── train.tsv    # ≈ 761 k 窗口（headerless）
    └── dev.tsv      # ≈ 190 k 窗口（headerless）
```

文件格式：

| 任务 | 列 1 | 列 2 | 表头 |
| --- | --- | --- | --- |
| 分类 | 4-mer 状态序列（空格分隔） | 整数 label (`0` / `1`) | 有（`sequence\tlabel`） |
| 回归 | 同上 | log-RPKM 浮点 | 无 |

`SeqLabelDataset` 根据 task 自动选择 label dtype；`has_header` 默认：classification=True，regression=False（可在 YAML `data.has_header` 覆盖）。

---

## 3. 配置文件

两份模板在 `configs/`：`ft_promoter_cls.yaml`、`ft_promoter_reg.yaml`。关键字段：

```yaml
task: classification | regression   # 决定 loss、head 维度、metric

tokenizer:
  type: kmer_cstate
  k: 4
  num_states: 15                    # 与 pretrain 一致

pretrained_ckpt: checkpoints/ch_promoter_tuned/epoch=0-val_loss=0.257.ckpt
classifier_dropout: 0.1 | 0.2       # paper 分类用 0.1，回归用 0.2

data:
  train_file: ...
  val_file: ...
  has_header: true | false
  batch_size: 32
  max_length: 512
  num_workers: 4
  max_train_samples: 80000          # 可选：大数据集抽样
  max_val_samples: 8000

model:
  hidden_size: 384                  # 必须与 backbone 完全一致
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 1536
  max_position_embeddings: 512
  hidden_dropout: 0.1
  attention_dropout: 0.1

optimizer:
  learning_rate: 2.0e-5             # legacy ChromBERT 分类 & 回归默认
  weight_decay: 0.01

scheduler:
  warmup_ratio: 0.1 | 0.2           # 分类 0.1，回归 0.2（对齐 paper）

trainer:
  max_epochs: 5 | 3
  precision: bf16-mixed
  accelerator: gpu
  devices: 1
  strategy: auto
  monitor: val_auc | val_pearson
  monitor_mode: max
  ckpt_dir: checkpoints/ft_promoter_cls
  log_name: ft_promoter_cls
  report_file: docs/runs/2026-04-18_ft_promoter_cls.json
```

`model.*` 与 backbone 的 shape 必须完全相同，否则 `load_state_dict` 的 `strict=False` 会“安静”地丢掉不匹配的权重。入口会打印 `missing / unexpected`：期望输出恰好是 head 的 4 个 key 作为 missing，`unexpected=[]`。

---

## 4. 启动训练

### 4.1 单卡（默认）

```bash
source ./activate.sh
CUDA_VISIBLE_DEVICES=0 fcbert-finetune --config configs/ft_promoter_cls.yaml
CUDA_VISIBLE_DEVICES=0 fcbert-finetune --config configs/ft_promoter_reg.yaml
```

单张 RTX 6000 Ada 上的实测时长（本次运行）：

| 任务 | 样本 | 每 epoch | 总时长 |
| --- | --- | --- | --- |
| 分类 | 20 k / 1 k | ≈ 17 s | ≈ 1.5 min（5 epoch） |
| 回归 | 80 k / 8 k | ≈ 60 s | ≈ 3 min（3 epoch） |

### 4.2 多卡（可选）

在 YAML 中把 `devices` 调大、`strategy: ddp`。微调轮数少，通常没必要，除非显著扩大 `max_train_samples`。

### 4.3 后台运行 + 日志

推荐写日志到 `logs/` 便于事后复盘：

```bash
nohup fcbert-finetune --config configs/ft_promoter_cls.yaml \
    > logs/ft_promoter_cls.log 2>&1 &
```

训练指标同时落在 `lightning_logs/<log_name>/version_*/metrics.csv`（由 `CSVLogger` 产出，可直接画收敛曲线）。

---

## 5. 指标口径

在 `src/flashchrombert/lightning/finetune.py:on_validation_epoch_end` 里计算：

### 分类（`num_labels=2`，CrossEntropyLoss）

- `val_acc` — argmax 准确率
- `val_f1` — sklearn `f1_score`（正类=1）
- `val_auc` — sklearn `roc_auc_score`（用 softmax 的 class-1 概率；若 dev 只有单一 label 则输出 NaN）

ckpt monitor 默认 `val_auc`，与 paper 对齐。

### 回归（`num_labels=1`，MSELoss）

- `val_mse` — 均方误差
- `val_pearson` — `scipy.stats.pearsonr`
- `val_spearman` — `scipy.stats.spearmanr`

ckpt monitor 默认 `val_pearson`。

### 指标聚合方式

Validation step 把 logits + labels 累积到 list，在 `on_validation_epoch_end` 一次性 concat 后计算——保证 AUC / Pearson 是 **全验证集**上的结果，而不是 batch-wise 平均。

---

## 6. 产出物

```
checkpoints/ft_promoter_{cls,reg}/
└── epoch={N}-val_{metric}={score}.ckpt      # save_top_k=1，按 monitor 取最佳

lightning_logs/ft_promoter_{cls,reg}/version_*/
├── metrics.csv                              # step/epoch 级指标
└── hparams.yaml

docs/runs/2026-04-18_ft_promoter_{cls,reg}.json   # 由 report_file 指向；包含 best_score, best_ckpt
```

`ckpt` 是 Lightning 格式（含 `state_dict` / `hyper_parameters` / `optimizer_states`）。若要加载用于推理：

```python
from flashchrombert.lightning import LitBertFinetune
lit = LitBertFinetune.load_from_checkpoint(
    "checkpoints/ft_promoter_cls/epoch=3-val_auc=0.7603.ckpt",
    config=...,  # 需要与训练一致的 BertConfig
)
```

---

## 7. 扩展：实现新的下游任务

重复工作主要集中在两层：

1. **数据层** — 若 TSV 形态就能表达任务，直接复用 `SeqLabelDataModule`，改 `task` 即可；若需要多字段或复杂 label，在 `src/flashchrombert/data/finetune.py` 里新增 Dataset。
2. **模型层** — 二分类/回归沿用 `BertForSequenceClassification`；其他头（多标签、token-level、对比学习）在 `src/flashchrombert/model/heads.py` 里新增，再写一个 `LitBert<Task>` 并在 `cli/finetune.py` 里 route task 名称。

新增 task 时：

- 监控指标要同步更新（`monitor`、`monitor_mode`、新指标的累积逻辑）。
- 若任务输出格式不同，新增的 pytest 要覆盖 `collate_fn` 的 label dtype；见 `tests/` 下已有的 MLM 测试作参考（本次微调 PR 暂未加单测，可在后续补充）。

---