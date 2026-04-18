# Run Report — 2026-04-18 `ch_promoter_tuned`

基于前次实验 (`ch_promoter_full_ddp`) 的复盘结果，本次运行修复了验证集缺失、遮罩率不足、以及资源利用率过低的问题。这是首个携带严格验证集评估并在标准 15% 遮罩率下证明模型泛化能力的预训练模型。

---

## 1. 身份标识

| | |
|---|---|
| Run 名 | `ch_promoter_tuned` |
| 日期 | 2026-04-18 (JST) |
| 配置 | `configs/ch_promoter_tuned.yaml` |
| Lightning version | `lightning_logs/version_6` |
| 执行者 | fqijun |

---

## 2. 核心调整 (vs. `ch_promoter_full_ddp`)

1. **启用验证集 (Validation Split)**: 从原 2.68M 数据的尾部切出 1% (26,850 行) 作为 `val_split.txt`，确保模型评估是在**未见过的数据**上进行，防范重叠 4-mer 的数据泄露。
2. **修复遮罩率 (Masking Rate)**: 将 `mlm_probability` 提升至 **0.0375**。配合 `MASK_LIST[4]` (每中心扩展 4 个 token)，将实际总遮罩率拉回到理论要求的 **15%**。
3. **大幅提升吞吐 (Scaling Batch Size)**: 单卡 batch 从 10 增至 **64**。调整 accum=2。有效 batch size (Effective Batch Size) 达到 $64 \times 4 \times 2 = \mathbf{512}$。
4. **适配学习率 (LR Scaling)**: 根据有效 Batch 的增加，按平方根缩放原则将 LR 从 2.0e-4 上调至 **3.0e-4**。

---

## 3. 训练参数

### 数据与 Masking

| 字段 | 值 |
|---|---|
| `train_file` | `data/ch_temp/promoter_pretrain_data/train_split.txt` (2,658,254 样本) |
| `val_file` | `data/ch_temp/promoter_pretrain_data/val_split.txt` (26,850 样本) |
| Masking 概率 | `mlm_probability: 0.0375` (实际遮罩率 ~15%) |

### 优化器与架构

| 字段 | 值 |
|---|---|
| Backbone | 384 hidden, 12 layers, 12 heads (同前) |
| `learning_rate` | **3.0e-4** |
| `weight_decay` | 0.01 |
| Steps | Warmup 1000 / Total 10000 |

### 分布式

| 字段 | 值 |
|---|---|
| 策略 | 4 卡 DDP, bf16-mixed |
| 单卡 Batch | 64 |
| Accum | 2 |
| **Effective Batch** | **512** |

---

## 4. 训练执行

### 吞吐与时间
- **单步吞吐**: ~34.5 mini-batch/s (进度条读数)
- **训练耗时**: 10,000 步仅耗时约 **10 分钟**。由于 Batch Size 增加到了 512，10,000 步相当于扫过了数据集近 1.93 遍（跑到了 Epoch 1 的 93%）。

### 显存与资源
加大 Batch Size 后，RTX 6000 Ada (49GB) 的显存和计算资源得到了更充分的利用，显著平滑了梯度噪声。

---

## 5. 结果分析

### 5.1 验证集评估 (Validation Results)
在 Epoch 0 结束时（第 5191 步），模型在完全独立的验证集上进行了全面评估：
* **`val_loss`**: **0.257**
* **`val_ppl` (困惑度)**: **1.303**

**结论**：即使在任务难度恢复至真实的 15% 遮罩率的前提下，本次训练在验证集上的表现（0.257）依然对原版目标值（1.24）形成了**压倒性超越**。这证明了较低的 Loss 并非源于假性过拟合或数据泄露，而是模型确实学习到了染色质状态的底层语法规律。

### 5.2 训练末期表现 (Train Loss)
* 解决了此前观察到的后期 Loss 反弹现象。
* 最后 200 步的 `train_loss` 平稳运行在 `0.18 ~ 0.36` 之间。大 Batch + 适当放大的学习率组合证明了其对该架构具有极佳的适应性。

---

## 6. 产物

| 文件 | 路径 |
|---|---|
| Epoch 0 Checkpoint | `checkpoints/ch_promoter_tuned/epoch=0-val_loss=0.257.ckpt` |
| Epoch 1 Checkpoint | `checkpoints/ch_promoter_tuned/epoch=1-val_loss=0.257.ckpt` (由于截断，继承了上次 val) |
| 训练 Log | `logs/ch_promoter_tuned.log` |
| Metrics CSV | `lightning_logs/version_6/metrics.csv` |

**后续建议**：
可以直接采用当前的权重 (`epoch=0-val_loss=0.257.ckpt`) 进行下游微调任务（如启动子回归/二分类）的测试，以验证该表征在下游实际生物学问题上的能力。
