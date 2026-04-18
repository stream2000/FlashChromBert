# Validation Plan — `ch_promoter_tuned` vs. Legacy ChromBERT

**日期**: 2026-04-18
**目标**: 用统一、可比、可量化的方法验证 FlashChromBert `ch_promoter_tuned` 相比
隔壁 legacy ChromBERT 的真实表现。替代掉隔壁目前那个仅靠 3 条手写序列打印
top-5 的 `zero_shot_mlm_test.py` 级别的"伪验证"。

---

## 1. 核心设计原则

1. **同数据 + 同遮罩 + 同 baseline**：双方脚本在同一份 val set、同一批遮罩位置、
   同一套 baseline 口径下评估，保证数值可直接对比。
2. **跨域泛化必测**：不能只看同分布 val，要加 OOD 数据集，证明模型不是靠
   memorize train 分布的 4-mer 得分。
3. **Baseline 必须存在**：解决"在单调背景下常数预测器也得高分"的问题。

## 2. 评估的数据集

| 代号 | 路径 | 条数 | 性质 |
|---|---|---|---|
| `promoter_val` | `data/ch_temp/promoter_pretrain_data/val_split.txt` | 26,850 | 同分布 (in-domain) |
| `crm_ood` | `data/ch_temp/crm_pretrain_data/crm_lim10_allcell_4merized.txt` | 37,464 | 跨域 — CRM 区域 |
| `genome_ood` | `data/ch_temp/genome_pretrain_data/pretrain_genome_all.txt` | 3,041 | 跨域 — 全基因组 |

三份数据双方项目共享（`data/ch_temp/` 结构完全一致）。

## 3. 遮罩策略（与预训练严格一致）

- `KmerMaskListMaskingStrategy(k=4, mlm_probability=0.0375)` — 与训练同口径
- 80/10/10 replacement (MASK / random / unchanged)
- **`torch.manual_seed(20260418)` 固定**，保证两边独立复现同一批遮罩位置
- `max_length = 512`

## 4. 指标

| 指标 | 含义 |
|---|---|
| `loss` | MLM CE loss（仅在 labels ≠ -100 位置） |
| `ppl` | `exp(loss)` |
| `top1_acc` | 被遮罩位置 top-1 预测准确率 |
| `top5_acc` | top-5 准确率 |

### Baseline（在完全相同的遮罩位置计算）

| Baseline | 预测规则 |
|---|---|
| `uniform` | 均匀分布 over vocab（非特殊 token） |
| `unigram` | 从 val set 统计的 token 频率 |
| `copy_left` | 复制左邻 token；若左邻是特殊/不可用则复制右邻 |

Baseline 的意义：`copy_left` 在 `AAAA AAAA [MASK] AAAA` 这种单调背景下
会拿到接近 100% 的 acc — 模型必须显著超过这一基线才能说明学到了东西。

## 5. 代码组织

```
FlashChromBert/
  src/flashchrombert/eval/
    __init__.py
    mlm.py              # 共享核心：metric 计算、baseline、dataset 加载
  scripts/
    eval_mlm.py         # FCB 端 runner，加载 Lightning ckpt

ChromBERT/training/examples/
  eval_mlm.py           # legacy HF 端 runner，加载 HF BertForMaskedLM
                        # (替代 zero_shot_mlm_test.py)
```

两边 runner 都复现同一套 metric / baseline / 遮罩逻辑；不共享 import
（跨 env 不现实），但通过**固定种子 + 同一份数据**保证数值可比。

## 6. 模型对照

| 代号 | 路径 | 备注 |
|---|---|---|
| `fcb_tuned` | `FlashChromBert/checkpoints/ch_promoter_tuned/epoch=0-val_loss=0.257.ckpt` | 我们的 |
| `legacy_promoter` | `data/ch_temp/promoter_pretrain_result/` | HF 格式 |
| `legacy_crm` | `data/ch_temp/crm_pretrain_result/` | HF 格式 |
| `legacy_genome` | `data/ch_temp/genome_pretrain_result/` | HF 格式 |

共享 vocab（5 特殊 + 15^4 = 50630），架构一致 (384/12/12/1536)，可横向比较。

## 7. 预期输出

`docs/runs/2026-04-18_validation_results.md`：一张 markdown 大表，
行 = 模型，列 = 数据集 × {loss, ppl, top1, top5}；独立三行列出 baseline。

## 8. 执行步骤

1. 实现 `src/flashchrombert/eval/mlm.py`（核心 metric + baseline）
2. 实现 `scripts/eval_mlm.py`（FCB runner）
3. 在 `flashchrombert` env 下跑 FCB 侧三个数据集
4. 实现 `ChromBERT/training/examples/eval_mlm.py`（legacy runner）
5. 在 `chrombert_training` env 下跑 legacy 侧三个数据集
   （promoter 模型跑三份数据；crm/genome 模型作为附加对照）
6. 汇总结果写入 `docs/runs/2026-04-18_validation_results.md`

## 9. 非目标（本次不做）

- 下游 finetune / linear probing（作为 follow-up，见 `crm_finetune_data/dev.tsv`）
- 替换 `sanity_check_mlm.py` / `strict_leak_check.py`（它们是 debug 工具，保留）
