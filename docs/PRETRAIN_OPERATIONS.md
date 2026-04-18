# FlashChromBert 预训练操作手册

本手册覆盖：环境、数据、配置、单卡/多卡启动、监控、断点续训、对齐原版 ChromBERT 的注意事项。

---

## 1. 环境

项目环境：conda env `flashchrombert`（Python 3.12，torch 2.5.1+cu124，Lightning 2.x）。

每次进入工作区：

```bash
cd /home/fqijun/python/FlashChromBert
source ./activate.sh
```

`activate.sh` 做两件事：
1. `conda activate flashchrombert`
2. 把 torch 自带的 `nvidia/*/lib` 预置到 `LD_LIBRARY_PATH`，避免 `libnvJitLink` 符号错误（详见根 `CLAUDE.md`）。

验证：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
# 期望：2.5.1+cu124 4
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

---

## 2. 数据

| 语料 | 路径 | 规模 | 用途 |
|---|---|---|---|
| 启动子 4-mer | `data/ch_temp/promoter_pretrain_data/pretrain_4mer_all.txt` | 2,685,105 行 / 346 MB | **默认**，对齐原版分发权重 |
| 全基因组 4-mer | `data/ch_temp/genome_pretrain_data/pretrain_genome_all.txt` | 3,041 行 / 3.1 GB | 长上下文变体 |
| CRM | `data/ch_temp/crm_pretrain_data/crm_lim10_allcell_4merized.txt` | 小 | 顺式调控模块子集 |

数据格式：一行一个样本，空格分隔的 4-mer，字母表 A–O（15 态）。`MLMDataset` 会把整个文件加载进内存（启动子 ~2 GB RAM，全基因组 ~12 GB RAM）。

---

## 3. 配置体系

所有配置在 `configs/`，YAML 六段：`tokenizer / masking / data / model / optimizer / scheduler / trainer`。

### 当前配置对照

| 文件 | 用途 | 模型 | 步数 | GPU |
|---|---|---|---|---|
| `tiny_text.yaml` | 字符级 demo | 128d × 4L | 2000 | 1 |
| `tiny_css.yaml` | 随机数据 smoke | 256d × 4L | 200 | 1 |
| `real_css_4mer.yaml` | k-mer 管线 smoke | 384d × 6L | 10 | 1 |
| `ch_promoter_smoke.yaml` | 真实数据 smoke | 384d × 6L | 100 | 1 |
| `ch_promoter_full_ddp.yaml` | **严格对齐原版 + 4 卡 DDP** | 384d × 12L | 10,000 | 4 |

### 对齐原版的关键参数（必须遵守）

这些从 `data/ch_temp/promoter_pretrain_result/{config.json,training_args.bin}` 反解得到：

- `model`: 12 层 / 384 hidden / 12 head / 1536 FFN / max_pos 512
- `tokenizer`: `kmer_cstate`, k=4, num_states=15 → vocab_size 50,630
- `masking`: `kmer_mask_list`, k=4, **mlm_probability 0.025**（**非** 0.15）
- `data`: `max_length 512`（**非** 256）
- `optimizer`: AdamW, lr=2e-4, weight_decay=0.01, betas=(0.9, 0.98)
- `scheduler`: 10,000 total_steps, 1,000 warmup
- 梯度裁剪：max_grad_norm=1.0
- 精度：bf16-mixed（原版 fp16 O1，我们更稳）

原版 eval loss 目标：`1.24`（见 `data/ch_temp/promoter_pretrain_result/eval_results.txt`）。

### DDP 上的 effective batch 说明

原版单卡 per-GPU batch=10 × grad_accum=5 → effective 50。
4 卡 DDP 保持同样 per-GPU batch=10 × accum=5 → effective **200**（×4）。

保守策略：**保持 LR=2e-4 不变**。这样等效 batch 变大但欠学习率——loss 下降更慢、更稳。
激进策略：按线性缩放 → LR=8e-4；按开方缩放 → LR=4e-4。第一次跑通不建议改。

---

## 4. 启动训练

### 单卡

```bash
CUDA_VISIBLE_DEVICES=0 fcbert-pretrain --config configs/ch_promoter_smoke.yaml
```

### 4 卡 DDP

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 fcbert-pretrain --config configs/ch_promoter_full_ddp.yaml
```

Lightning 在 `strategy=ddp + devices=4` 时会自行 fork 4 个子进程，无需 `torchrun`。日志和进度条由 rank 0 输出。

### 后台运行

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    fcbert-pretrain --config configs/ch_promoter_full_ddp.yaml \
    > logs/ch_promoter_full_ddp.log 2>&1 &
echo $! > logs/ch_promoter_full_ddp.pid
```

---

## 5. 监控

### GPU 占用

```bash
watch -n 2 nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
```

### 训练日志

- Lightning CSV logger：`lightning_logs/version_N/metrics.csv`（每 `log_every_n_steps` 写一行）
- stdout：进度条 + loss

提取训练曲线：

```bash
python - <<'PY'
import pandas as pd
df = pd.read_csv("lightning_logs/version_<N>/metrics.csv")
print(df[["step","train_loss_step","train_loss_epoch"]].tail(20))
PY
```

### Checkpoint

- 路径：`checkpoints/<run_name>/`（由 `trainer.ckpt_dir` 决定）
- 命名：`epoch={e}-val_loss={v:.3f}.ckpt`（无 val 时 `v=0.000` 占位）
- 保留：`save_top_k=3`

---

## 6. 断点续训

Lightning 自动保存优化器与 scheduler 状态。手动恢复：

```python
trainer.fit(lit_model, dm, ckpt_path="checkpoints/ch_promoter_full_ddp/last.ckpt")
```

当前 CLI 暂不接受 `--resume`；如需常态化需在 `pretrain.py` 加个 flag。一次性续训，直接在 Python 里调 `trainer.fit(..., ckpt_path=...)`。

---

## 7. 性能预期（4× RTX 6000 Ada，bf16，seq=512）

| 配置 | 吞吐 | 10k 步耗时 |
|---|---|---|
| 单卡 6 层 seq=256 | 43 step/s | — |
| 单卡 12 层 seq=512 | ≈ 7 step/s（含 accum=5 → 1.4 optimizer step/s） | ~2 h |
| **4 卡 12 层 seq=512** | ≈ 25 step/s / optimizer-step 5 | **~35 min** |

第一个 epoch 前约 1–2 分钟额外开销：一次性把 2.68M 行加载进内存 + 构建 tokenizer。

---

## 8. 与原版 ChromBERT 的差异（预期）

| 维度 | 原版 | FlashChromBert |
|---|---|---|
| 注意力内核 | HF eager | SDPA / FA2（SM86+ 自动启用） |
| 精度 | fp16 O1（loss scaler） | bf16-mixed（无需 scaler） |
| 分布式 | 单卡 | 支持 4 卡 DDP |
| 代码栈 | HF Trainer + 自定义脚本 | Lightning + 精简 CLI |
| 配置 | argparse | YAML（可版本化） |

应有的 **等价性**：模型结构、tokenizer vocab、masking 方案、LR schedule。

---

## 9. 故障排查

- `libnvJitLink` 符号错误 → 漏 `source ./activate.sh`。
- `CUDA OOM` → 降 `per_gpu_train_batch_size` 或提高 `accumulate_grad_batches`；不要降 `max_length`（会破坏原版等价）。
- DDP 挂死 / NCCL 超时 → 检查 `CUDA_VISIBLE_DEVICES` 与 `devices` 是否匹配；多卡第一次运行可加 `NCCL_DEBUG=INFO`。
- `RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types` → 检查 tokenizer vocab 是否一致（vocab_size 必须 50,630）。
- Lightning 输出 `Total length of list across ranks is zero` → val_dataloader 返回了 `[]`，属正常，无 val 时的警告。

---

## 10. 对齐原版复现路径

1. 跑 smoke（已完成）：`configs/ch_promoter_smoke.yaml`，确认管线贯通。
2. 跑 full DDP：`configs/ch_promoter_full_ddp.yaml`，10k 步。
3. 收敛后比对 `checkpoints/ch_promoter_full_ddp/*.ckpt` 的 MLM loss 与 `data/ch_temp/promoter_pretrain_result/eval_results.txt` 末端值（1.24）。
4. 若显著高于 1.24，按以下顺序排查：
   - mlm_probability 是否确为 0.025
   - max_length 是否确为 512
   - vocab_size 是否确为 50,630
   - total_steps / warmup 是否匹配 LR schedule
5. 若要一步到位等价，考虑直接加载原版 `pytorch_model.bin` 到本项目 `BertForMaskedLM` 里做零样本对齐测试（另行实现）。
