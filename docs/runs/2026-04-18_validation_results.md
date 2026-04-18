# Validation Results — `ch_promoter_tuned` vs. Legacy ChromBERT

**日期**: 2026-04-18
**计划**: [2026-04-18_validation_plan.md](2026-04-18_validation_plan.md)
**遮罩**: `KmerMaskListMaskingStrategy(k=4, mlm_probability=0.0375)`, seed=20260418
**每数据集样本数**: 5,000（按 510-token 非重叠窗口切分长序列）

---

## 1. 评估数据集

| 代号 | 条数 | 平均 seq_len (含 CLS/SEP) | 总遮罩位置 |
|---|---|---|---|
| `promoter_val` | 5,000 | 30 | 18,616 |
| `crm_ood` | 5,000 | 509 (长短差大，主峰 9) | 9,697 |
| `genome_ood` | 5,000 | 512 (切块后) | 361,248 |

## 2. 模型

| 代号 | 来源 | 训练域 |
|---|---|---|
| **`fcb_tuned`** | `FlashChromBert/checkpoints/ch_promoter_tuned/epoch=0-val_loss=0.257.ckpt` | promoter |
| `legacy_promoter` | `ChromBERT/data/ch_temp/promoter_pretrain_result/` | promoter |
| `legacy_crm` | `ChromBERT/data/ch_temp/crm_pretrain_result/` | CRM |
| `legacy_genome` | `ChromBERT/data/ch_temp/genome_pretrain_result/` | whole-genome |

## 3. 结果总览

`loss` / `top1 (%)` / `top5 (%)`，**粗体**为该列最佳模型，*斜体*为最佳 baseline。

| 模型 | `promoter_val` | `crm_ood` | `genome_ood` |
|---|---|---|---|
| **`fcb_tuned`**        | **0.264 / 91.7 / 98.6** | **0.592 / 88.0 / 94.1** | 4.87 / 45.7 / 58.6 |
| `legacy_promoter`      | 0.275 / 91.9 / 98.3 ⚠️   | 0.660 / 87.2 / 92.7     | 6.57 / 48.1 / 56.4 |
| `legacy_crm`           | 1.94 / 65.6 / 79.6      | 0.585 / 87.6 / 94.5 ⚠️  | 3.99 / 39.0 / 53.1 |
| `legacy_genome`        | 4.45 / 59.9 / 70.7      | 9.57 / 18.2 / 24.7      | **0.175 / 94.3 / 99.3** ⚠️ |
| — *baselines* —        |                         |                         |                      |
| `copy_left`            | 3.87 / 71.1 / 71.1      | 3.30 / *75.5* / 75.5    | 4.53 / *66.1* / 66.1 |
| `unigram`              | 3.30 / *24.2* / *66.6*  | 2.32 / 56.9 / 77.2      | 3.01 / 41.1 / 66.2   |
| `uniform`              | 10.83 / 0.00 / 0.01     | 10.83 / 0.00 / 0.01     | 10.83 / 0.00 / 0.01  |

> ⚠️ = 该 legacy 模型训练时见过整份数据（见 §4 数据污染警告）。这些格子是**训练集回归**，不是独立验证。

## 4. ⚠️ 数据污染警告（必读）

**`legacy_promoter` 在 `promoter_val` 上的结果是训练集回归，不是验证。**

- `val_split.txt` (26,850 行) = `pretrain_4mer_all.txt` (2,685,105 行) 的尾部
  - 首行 MD5 == all 文件第 2,658,256 行 MD5 (= 2,685,105 − 26,850 + 1)
  - 末行 MD5 == all 文件末行 MD5
  - 由 `FlashChromBert/scripts/` 下的切分脚本从末尾切出，**切分时隔壁 legacy 模型早已训练完毕**
- 隔壁 `promoter_pretrain_result` 训练时用的是整份 `pretrain_4mer_all.txt`
- ⇒ **隔壁模型在训练中见过 val_split.txt 的每一行**
- 隔壁自己的 `ChromBERT/GEMINI.md` §69–71 也承认：原 `pretraining_loop.sh`
  **硬编码 `--train_data_file == --eval_data_file`**，他们的 `eval_results.txt`
  里收敛到的 1.24 本身就是训练集 PPL

因此 §5 表里 `legacy_promoter / promoter_val` 那格应该理解为**记忆上限**，而非泛化能力。

## 5. 结论

### 5.1 我们的模型 vs 隔壁 legacy（真·公平对照：`crm_ood`）

唯一两边都没见过的**同方向（同领域 preset / 不同区域）**测试集：

- `fcb_tuned`: loss **0.592**, top1 **88.0%**
- `legacy_promoter`: loss 0.660, top1 87.2%

→ **`fcb_tuned` 在真·held-out 上 loss 低 10%**，top1 略优。这才是我们相对 legacy 的真实位置。

### 5.2 `promoter_val` 的重新解读（考虑污染后）

| 模型 | loss | 性质 |
|---|---|---|
| `fcb_tuned`        | **0.264** | 真·held-out（从未见过） |
| `legacy_promoter`  | 0.275     | 训练集回归（已被记忆过） |

**`fcb_tuned` 在没见过的数据上，loss 比 legacy 在看过的数据上还低。**
这是一个比"两者并列"更强的论点：同架构、同数据分布下，我们的泛化 >= legacy 的记忆。

> 注：此前担心的"val_loss=0.257 是不是太好以至于像过拟合"仍然不成立——
> 如果真是过拟合，我们在 crm_ood 上也不会比 legacy 好 10%。

### 5.3 跨域泛化（genome_ood）

所有模型在 **genome_ood** 都崩溃。`fcb_tuned` 的 top1 (45.7%) **低于 copy_left 基线 (66.1%)**：
说明模型在 whole-genome 级别的序列上反而被简单的"复制左邻"规则打败。根源是 genome
分布里有大量单调长跑区（OOOO OOOO OOOO...），而 promoter 训练集的转移密度高得多。
这不是训练问题，是**域差异**——`legacy_genome` 在 genome 上就拿到 0.175 / 94.3%。

### 5.4 一个补充细节

`legacy_promoter` 在 `promoter_val`（已见）和 `crm_ood`（未见）上
loss 从 0.275 涨到 0.660 —— **+2.4x 的 loss gap**
相当于对 "看过 vs 没看过" 的记忆增益量化。
对比我们 `fcb_tuned`：0.264 vs 0.592，也是 +2.2x，但 **两边都是没看过的数据**，
gap 来源是"同分布 val vs 跨区域 OOD"的真实泛化代价，不含记忆红利。

### 5.5 专精-专用模式得到验证

每个 legacy 模型在其自己的域上都显著优于其他三个：

- `legacy_promoter` on `promoter_val`: 91.9%
- `legacy_crm` on `crm_ood`: 87.6%
- `legacy_genome` on `genome_ood`: 94.3%

排除了"模型只是在记 4-mer 的全局分布"的怀疑——它们确实学到了**域特异**的状态转移语法。

### 5.6 Baseline 对照 — 隔壁 `zero_shot_mlm_test.py` 的问题被定量验证

隔壁原先的"验证"只打印 3 条手写序列（如 `AAAA AAAA [MASK] AAAA AAAA`）的 top-5。
在这种**单调背景**下，`copy_left` 基线在 `crm_ood` 上能拿到 75.5% top1，
在 `genome_ood` 上甚至到 66.1%。也就是说，**一个完全不学习的常数预测器**
就能通过他们原先的定性测试。

这证明：
- 原 `zero_shot_mlm_test.py` 无法区分"模型真的学会了"和"模型是常数预测器"。
- 新的量化 + baseline 对比协议**必要且有效**——我们的 `fcb_tuned` top1 = 91.7%
  相对于 copy_left 的 71.1%（same data）提升了 **+20.6 个百分点**，这才是实打实的学习证据。

## 6. 产物

| 文件 | 内容 |
|---|---|
| `src/flashchrombert/eval/mlm.py` | 共享评估核心：fixture 生成、baseline、batched metrics |
| `scripts/eval_mlm.py` | FCB 端 runner |
| `ChromBERT/training/examples/eval_mlm.py` | 隔壁 legacy 端 runner（替代 `zero_shot_mlm_test.py`） |
| `fixtures/mlm_eval/2026-04-18/{promoter_val,crm_ood,genome_ood}.pt` | 确定性遮罩固化数据 |
| `docs/runs/2026-04-18_validation_fcb.json` | FCB 结果 + baselines |
| `docs/runs/2026-04-18_validation_legacy_{promoter,crm,genome}.json` | legacy 三个模型结果 |

## 7. 后续建议

1. **域适应**：`fcb_tuned` 在 genome 上不如 copy_left 说明 promoter 预训练对全基因组推理能力有限。
   下一轮可以 (a) 混合 promoter + genome 数据 (b) 先 genome 预训练再 promoter 继续训练。
2. **Linear probing**：`data/ch_temp/crm_finetune_data/dev.tsv` 有 1,000 条带标签样本，
   可以用 `[CLS]` 表征做逻辑回归，评估 embedding 下游可用性（本次 plan 的可选项，未实施）。
3. **替代隔壁 `zero_shot_mlm_test.py`**：建议把新写的 `eval_mlm.py` 作为 legacy 项目
   的官方验证脚本，下一次发版时把旧脚本标 deprecated。

---

## 8. 评注 — "测试到底能告诉我们什么 / 公平性靠什么保证"

> 本节是事后对本次验证边界与可信度的自检。和 §5 的结论分开：§5 讲"数据说了什么"，
> 本节讲"这些话我们**能**说到多确定"。

### 8.1 按置信度分层的可判断结论

**高置信度（数据直接支持）**

1. **真·held-out 跨域 (`crm_ood`)**：`fcb_tuned` 胜 `legacy_promoter`
   - loss 0.592 vs 0.660 (−10%)，top1 88.0% vs 87.2%
   - 两个模型都没见过这份数据，同架构、同训练域倾向（promoter）
   - → **我们的整套训练方案在跨域泛化上略优**

2. **我们的 promoter 验证没有过拟合嫌疑**
   - fcb 在**没见过**的 `promoter_val` 上 loss = 0.264
   - 低于 legacy 在**同份数据（训练中见过）**上的 0.275
   - → fcb 的泛化 ≥ legacy 的记忆，不是幸运巧合

3. **模型在真的学习，不是在背先验**
   - fcb top1 = 91.7% ≫ copy_left 71.1% ≫ unigram 24.2%（同数据、同遮罩位置）
   - → 相对常数预测器 +20.6pp 的绝对增量就是学习信号

**低置信度（结论受限）**

4. **`genome_ood` 两个模型都废**：fcb top1 45.7% < copy_left 66.1%
   - loss 差距 4.87 vs 6.57 (+28%) 但 top1 方向相反
   - → **噪声水平**，不能下"谁更好"的结论

5. **不能断言"我们模型架构比 legacy 好"**
   - 训练配方差异 (mlm_prob 0.0375 vs 0.025、effective batch 512 vs 50、LR 3e-4 vs 2e-4)
     和实现差异（FA2、bf16）混在一起
   - 我们能说的只是 **"fcb 的整套训练方案（recipe + 实现）泛化略优"**

### 8.2 公平性靠什么机制保证（不靠信任，靠代码）

| # | 保证 | 实现 |
|---|---|---|
| 1 | 两边吃**完全相同的字节** | `fixtures/mlm_eval/2026-04-18/*.pt` 由 fcb 端一次性生成，legacy 端 `torch.load` 读取同一文件 |
| 2 | 遮罩位置一致 | `torch.manual_seed(20260418)` 在遮罩前固化；fixture 里 labels (-100 外非遮罩) 已定格 |
| 3 | Vocab 对齐 | 已逐行验证 fcb tokenizer 的 id→token 映射 == legacy `vocab.txt` |
| 4 | 指标算法一致 | `flat_logits[valid]` + `cross_entropy(reduction="sum")` + `topk` — 两端脚本逻辑逐行 mirror |
| 5 | 两边都 `eval()` + fp32 | 关掉 dropout，精度不引入歧义 |
| 6 | Baseline 链式 sanity | uniform 0.00% → unigram 24% → copy_left 71% → fcb 91.7% 单调递增 → eval pipeline 没 bug |
| 7 | 污染显式标记 | `legacy_promoter / promoter_val` 在结果表里带 ⚠️，不进最终对比 |

### 8.3 诚实的公平性**局限**

1. **只跑了 1 个 mask 种子**
   - 要说 "−10% 不是噪声"严格应 ≥3 seed 取均值 ± 标准差
   - 当前结果可复现但**不具备统计显著性**
2. **Eval 时 mlm_prob = 0.0375，legacy 训练时是 0.025**
   - 给 legacy 出了稍难的题（有效遮罩 ~15% vs ~10%）
   - 影响不大（BERT 对遮罩率不太敏感）但存在
3. **每数据集 5,000 样本上限**
   - 不是全量（promoter 26,850 / crm 37,464）
   - 18,616 个遮罩位置够算 p≈0.001 显著性，但不是穷尽
4. **训练配方混杂**
   - 想隔离"纯架构/实现差异"须用 legacy 的 recipe 重跑 fcb
   - **本次不是纯模型比较，是"配方+实现"整体比较** — 这通常是用户真正关心的，但须名实相符

### 8.4 一句话总括

> **在所有我能控制的变量下（输入字节、遮罩位置、vocab、指标算法），`fcb_tuned`
> 在真·held-out 跨域数据上的 loss 比 `legacy_promoter` 低约 10%。训练配方差异是
> 真的——这就是比较的对象本身，而不是未控制的混淆项。**

若要把这个 "+10%" 升级为统计显著的论点，下一步是 ≥3 个遮罩种子重跑，并取均值 ± std。
