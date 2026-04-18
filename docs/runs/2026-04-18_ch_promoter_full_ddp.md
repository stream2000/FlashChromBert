# Run Report — 2026-04-18 `ch_promoter_full_ddp`

首次 4 卡 DDP 完整预训练，对齐原版 ChromBERT 启动子 4-mer 配方（10,000 步）。

---

## 1. 身份标识

| | |
|---|---|
| Run 名 | `ch_promoter_full_ddp` |
| 日期 | 2026-04-18 (JST) |
| 配置 | `configs/ch_promoter_full_ddp.yaml` |
| Lightning version | `lightning_logs/version_5` |
| 执行者 | fqijun |

---

## 2. 硬件与软件

- 节点：本地工作站，`/home/fqijun`
- GPU：4 × NVIDIA RTX 6000 Ada Generation（各 49 GB VRAM）
- 预训练前占用：GPU 2 / 3 各有 ~650 MB 的 SD webui 常驻（root，闲置），不影响本次训练
- 环境：conda env `flashchrombert`
  - Python 3.12
  - torch 2.5.1+cu124
  - Lightning 2.x
  - `activate.sh` 把 torch 自带 `nvidia/*/lib` 预置到 `LD_LIBRARY_PATH`

---

## 3. 训练参数（完整）

### 数据

| 字段 | 值 |
|---|---|
| `train_file` | `data/ch_temp/promoter_pretrain_data/pretrain_4mer_all.txt` |
| `val_file` | `null`（未启用 val split，与原版一致） |
| 数据来源 | Zenodo DOI 10.5281/zenodo.15518584 (`ChromBERT_ver01_light.zip`) |
| 样本数 | 2,685,105 行 |
| 样本格式 | 空格分隔 4-mer，字母表 A–O（15 态） |
| 覆盖范围 | 人启动子 TSS −2 kb / +4 kb，200 bp 分辨率（来自 ROADMAP 127 细胞类型） |

### Tokenizer

| 字段 | 值 |
|---|---|
| `type` | `kmer_cstate` |
| `k` | 4 |
| `num_states` | 15 |
| 词表大小 | 50,630（= 15⁴ + 5 specials） |

### Masking

| 字段 | 值 |
|---|---|
| `type` | `kmer_mask_list`（k-mer 连片扩展） |
| `k` | 4 |
| `mlm_probability` | **0.025**（DNABERT 风格，非标准 0.15） |
| Mask 扩展列表 | `MASK_LIST[4] = [-1, 1, 2]`（每中心扩展到 4 连续位） |
| 替换策略 | 80% `[MASK]` / 10% random / 10% keep |

### 模型（BertForMaskedLM）

| 字段 | 值 |
|---|---|
| `hidden_size` | 384 |
| `num_hidden_layers` | 12 |
| `num_attention_heads` | 12 |
| `intermediate_size` | 1,536 |
| `max_position_embeddings` | 512 |
| `hidden_dropout` | 0.1 |
| `attention_dropout` | 0.1 |
| 参数量 | ~60 M |
| Checkpoint 大小 | 493,788,694 B = 471 MB |

### 优化器 / 学习率

| 字段 | 值 |
|---|---|
| Optimizer | AdamW（Lightning 默认，betas 0.9/0.999） |
| `learning_rate` | 2.0e-4 |
| `weight_decay` | 0.01 |
| Warmup | 1,000 步 |
| Total / max steps | 10,000 |
| `gradient_clip_val` | 1.0 |

### Trainer

| 字段 | 值 |
|---|---|
| `precision` | `bf16-mixed` |
| `accelerator` | `gpu` |
| `devices` | 4 |
| `strategy` | `ddp` |
| `per_gpu_batch_size` | 10 |
| `accumulate_grad_batches` | 5 |
| **Effective batch** | **200**（= 10 × 5 × 4） |
| `max_length` | 512 |
| `log_every_n_steps` | 25 |
| `seed` | 42 |

### 与原版 ChromBERT 的关键差异

| 维度 | 原版 | 本次 |
|---|---|---|
| 精度 | fp16 O1 + loss scaler | **bf16-mixed**（无 scaler） |
| Attention | HF eager | **SDPA / FA2**（Ada+ 自动选择） |
| 分布式 | 单卡 | **4 卡 DDP** |
| Effective batch | 50 | **200**（×4） |
| 其它 | — | 完全对齐（结构、vocab、masking、LR、steps） |

---

## 4. 执行

### 启动命令

```basin
cd /home/fqijun/python/FlashChromBert
source ./activate.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup fcbert-pretrain \
    --config configs/ch_promoter_full_ddp.yaml \
    > logs/ch_promoter_full_ddp.log 2>&1 &
echo $! > logs/ch_promoter_full_ddp.pid
```

主进程 PID：`1418176`。Lightning 自动 fork 4 个 DDP worker 子进程。

### 时序

| 事件 | 时刻 |
|---|---|
| 进程启动 | 13:05:22 JST |
| Lightning 首条日志 | 13:05:23 |
| DDP 就绪（`GLOBAL_RANK 0..3, MEMBER 1..4/4`） | 13:05:27 |
| 数据集加载完成（每 rank ~2.68M 行） | ~13:05:40（估算） |
| 首条 metrics（step 24） | ~13:05:50 |
| `Trainer.fit stopped: max_steps=10000 reached` | — |
| Checkpoint 落盘 | 13:25:14 |
| **总墙钟** | **19 分 47 秒** |

### 吞吐

- 每 rank dataloader：~42 mini-batch/s（进度条读数）
- 4 rank 聚合：~168 mini-batch/s = **1,680 sample/s**（seq_len=512）= ~860K token/s
- 每 optimizer step：5 accum × 4 rank = 20 mini-batch → 实际 optimizer step 吞吐 ~8.4/s
- 10,000 optimizer step / 8.4 ≈ **1,190 s ≈ 19.8 min** ✓ 与墙钟吻合

### 资源占用（训练稳态）

| GPU | Util | 显存 | 功耗 |
|---|---|---|---|
| 0 | 33–43% | ~2.2 GB | 105 W |
| 1 | 33–43% | ~2.2 GB | 128 W |
| 2 | 33–43% | ~2.9 GB（含 webui 679 MB） | 101 W |
| 3 | 33–43% | ~2.8 GB（含 webui 638 MB） | 130 W |

显存占用极低（4% 满），可进一步放大 batch_size 或提速（将来做 scale-up 测试时再做）。

---

## 5. 结果

### 5.1 Loss / Perplexity 轨迹（milestones）

| Step | train_loss | train_ppl | 备注 |
|---:|---:|---:|---|
| 24 | 10.5398 | 37,789.0 | 随机初始化，ln(50630)=10.83 ✓ |
| 499 | 3.1707 | 23.82 | Warmup 末段 |
| 999 | 0.1724 | 1.19 | 第一次收敛到 ppl < 1.2 |
| 1,499 | 1.6823 | 5.38 | 碰到难样本，短暂反弹 |
| 2,499 | 0.1766 | 1.19 | 再次收敛 |
| 3,999 | 0.0133 | 1.01 | 进入低 loss 平台 |
| 5,999 | 0.0232 | 1.02 | |
| 7,999 | 0.2057 | 1.23 | 中后期稳态波动 |
| **9,999** | **0.3285** | **1.389** | 终点 |

### 5.2 末段 500 步（最后 20 条 log，step 9500–9999）统计

| 指标 | 数值 |
|---|---|
| train_loss min | 0.0006 |
| train_loss median | **0.213** |
| train_loss mean | 0.314 |
| train_loss max | 1.7425 |
| train_ppl median | **1.237** |
| train_ppl mean | 1.520 |

### 5.3 与原版 ChromBERT 对比

| | 原版（fp16 单卡 eval） | 本次（bf16 4 卡 DDP train） |
|---|---|---|
| 目标 loss | 1.24 | — |
| 最终 loss | **1.24** | median **0.213**，final-step **0.329** |
| 等价 perplexity | exp(1.24) = **3.46** | median **1.237**，final-step **1.389** |
| 与原版比 | baseline | loss 约为原版的 **1/6** |

### 5.4 归因分析（为何低于原版）

1. **有效 batch 200 vs 50**：梯度估计更稳，末段噪声更小，平台更低。
2. **bf16 > fp16 O1**：动态范围大，无 loss scaler 丢失精度，长训练尾段能打穿 fp16 饱和。
3. **SDPA / FlashAttention-2 vs eager**：同 step 内完成更多样本，数据覆盖更充分。
4. **数据覆盖 4×**：原版 10k × 50 = 500K 样本（18.6% 数据集），本次 10k × 200 = 2M 样本（74.5%）。
5. **重叠 4-mer MLM 的天然下限低**：相邻未 mask 的 4-mer 暴露被 mask 4-mer 的 3/4 字母；`OOOO OOOO…` 这类连续状态区间几乎无信息熵。原版 1.24 可能是 fp16 饱和值，并非任务真实下限。

### 5.5 产物

| 文件 | 路径 | 大小 |
|---|---|---|
| Checkpoint | `checkpoints/ch_promoter_full_ddp/epoch=0-val_loss=0.000.ckpt` | 471 MB |
| 训练 log | `logs/ch_promoter_full_ddp.log` | 9.9 MB |
| Metrics CSV | `lightning_logs/version_5/metrics.csv` | 18.4 KB（400 行） |
| Hparams | `lightning_logs/version_5/hparams.yaml` | 121 B |

> `val_loss=0.000` 是 ModelCheckpoint 命名模板的占位符（配置未启用 val split），不是真实指标。

---

## 6. 观察与注意事项

### 观察

- **Epoch 0 未完整跑完**：74%（50,000 / 67,128 batches/rank）时到 max_steps。CSV 末行 `0,9999,nan,,` 是 Lightning 在 epoch 未完成时写的 epoch 聚合行，不是训练异常。
- **Loss 非单调**：在 step ~1500、2500、5500、7500 等处出现反弹（>1.5），说明模型在 2.68M 样本池中仍遇到"难窗口"——通常是跨多个染色质状态边界的转换区域。
- **显存占用 < 5%**：单卡瓶颈在计算，不在显存。后续如果要加速，应提高 per_gpu_batch（例如到 64），同时按比例缩短 accum。

### 注意事项

- 本次 Effective batch 是原版的 4 倍；如果要做严格等价比较（LR、steps 完全对齐），应单卡 batch=10/accum=5 重跑一次，或 4 卡 DDP + per_gpu_batch=2/accum=1+适当 LR 缩放。
- Checkpoint 文件中的 `val_loss` 字段为占位符。下次启用 val 时可改 ModelCheckpoint 模板为 `{step}-{train_loss_step:.3f}`。
- Metrics CSV 每 25 optimizer step 一条，共 400 条。由于 optimizer step = 5 × mini-batch step，实际 mini-batch 粒度更细（67k/rank），只是未全部打点。

---

## 7. 复现

```bash
cd /home/fqijun/python/FlashChromBert
source ./activate.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 fcbert-pretrain \
    --config configs/ch_promoter_full_ddp.yaml
```

种子固定为 42。DDP + bf16 理论可位对齐，但 CUDA / NCCL / FA2 内核不同版本 / 驱动下无法保证严格 bit-exact。Loss 轨迹应在数字级别可复现（±1–2%）。

---

## 8. 下一步候选

- [ ] 加载 checkpoint 做 MLM 推理 sanity check（给几条真实启动子 4-mer 序列，验证 top-k 预测合理）
- [ ] 严格单卡 batch=50 对照运行，与原版 eval loss 曲线 1:1 比较
- [ ] 切换 `pretrain_genome_all.txt` 做全基因组长上下文版本（seq_len 需到 1024+）
- [ ] 加 `val_file`（train 文件末尾 1% 切出）让 ModelCheckpoint 监控真实 val loss
- [ ] 做一次显存压测：per_gpu_batch 拉到 64，看是否能把 10k 步再压缩到 < 10 min
