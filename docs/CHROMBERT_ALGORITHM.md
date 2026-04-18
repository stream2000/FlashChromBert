# ChromBERT 关键预处理与预训练算法摘要 (更新版)

本文档总结了 ChromBERT 核心算法流程，并结合对原始源码库（Legacy）与现代化复现（FlashChromBert）的对比研究，深入解析了其训练细节与工程实现。

## 1. 数据预处理 (Data Preprocessing)

*   **字母表映射 (Alphabetical Encoding)**：将 ChromHMM 的数字状态 (1-15 或 1-18) 映射为英文字母 (A-O 或 A-R)。
*   **K-mer Tokenization**：采用连续重叠切分。由于词表随 $k$ 爆炸，最终选择 **4-mer**（15态词表 50,630；18态词表 104,980）。
*   **双版本数据集**：
    *   **Promoter (启动子版)**：基于 ROADMAP 15态，仅覆盖 TSS 上下文，文本量约 **2.8 GB** (268万样本)。
    *   **Whole Genome (全基因组版)**：基于 IHEC 18态，覆盖整条染色体，文本量约 **41.5 GB**。

## 2. 预训练算法与工程实现 (Pretraining & Engineering)

预训练的核心在于掩码语言模型 (MLM)，但其实际执行在不同阶段有所不同：

### 2.1 训练循环策略 (Epochs vs. Steps)
*   **预训练 (Pretraining)**：是以 **Steps (步数)** 为终点的任务。论文与代码默认设置为 **10,000 steps**。
    *   由于数据量巨大，10,000 步通常远不足以跑完 1 个完整的 Epoch。
    *   在原始 Promoter 训练中，10,000 步仅覆盖了约 18.6% 的数据；而在 4 卡 DDP 复现中，由于有效 Batch 放大，10,000 步可覆盖约 74.5% 的数据。
*   **微调 (Fine-tuning)**：遵循标准深度学习习惯，采用 **Epoch (轮数)** 驱动，默认通常为 **10 epochs**。

### 2.2 硬件妥协与“切块循环” (Split-Chunk Strategy)
*   **原始实现 (Legacy)**：受限于 2080Ti (11GB) 显存和有限的内存，原作者无法一次性加载 41GB 数据。
    *   **方案**：将数据切分为多个 Chunks，通过 Bash 脚本进行 `for` 循环轮询训练。每跑完一个 chunk (约 5000 步) 就保存一次权重，并作为下一块的起点。
*   **现代实现 (FlashChromBert)**：利用大规模显存 (RTX 6000 Ada 48GB) 和现代化 DDP 策略。
    *   **优化**：不再需要打断式的切块训练，可直接进行全量数据加载或通过 `IterableDataset` 流式读取，保证了数据混洗 (Shuffle) 的全局性。

### 2.3 精度与优化器
*   **混合精度 Bug**：原生代码虽使用了 Apex FP16，但由于在 `run_pretrain.py` 中遗漏了 `amp.scale_loss` 上下文管理，导致 Loss Scaling 实际上未生效，易在训练尾段出现梯度下溢。
*   **BF16 优势**：在现代化复现中改用 `bf16-mixed`，利用 Ada 架构原生支持，无需 Loss Scaling 即可获得比原生 FP16 更稳定的收敛轨迹（Loss 降至 0.3 级别，而原版在 1.2 左右饱和）。

## 3. 下游分析：Motif 提取与聚类算法

*   **注意力图谱**：利用 BERT 的多头注意力机制，识别对分类或回归贡献最大的“高贡献区间”。
*   **动态时间规整 (DTW)**：关键算法创新。解决了染色质状态序列长度可变、动态漂移的问题，能够计算不同长度 Motif 之间的结构相似度。
*   **凝聚层次聚类**：基于 DTW 距离矩阵，将发现的 Motif 自动归类为生物学功能类似的家族。

## 4. 训练规模对比 (Summary)

| 场景 | 有效 Batch Size | 训练终止条件 | 核心挑战 |
| :--- | :--- | :--- | :--- |
| **原生论文 (2080Ti)** | 50 (5 batch * 10 accum * 1卡) | 10,000 steps | 显存极小，无法 FlashAttention |
| **现代化复现 (4卡 Ada)** | 200 (10 batch * 5 accum * 4卡) | 10,000 steps | 速度极快，Loss 平台期更低 |
| **全基因组训练** | 变量 | 分块循环 + Early Stop | 41GB 数据 IO 与内存瓶颈 |

## 5. 启动子模型核心参数配置与实验指标 (Promoter Model Configuration)

基于原始论文与项目文档，启动子版本模型（Promoter Model）的具体架构与任务设置如下：

### 5.1 模型基础架构 (Backbone Architecture)
与 DNABERT 的大架构范式类似，但**参数规模减半**（在论文的方法部分专门明确提及）：
*   **Hidden Size**: 384
*   **Intermediate Size**: 1,536
*   **Num Hidden Layers**: 12
*   **Num Attention Heads**: 12
*   **Max Position Embeddings**: 512
*   **Dropout (Hidden & Attention)**: 0.1
*   *注记*：论文行文中有几处疑似笔误（如早期提及 DNABERT 的 768），但在方法论核心段落详细规定“hidden size of 384. The intermediate layer size was 1536”。项目中实验配置文件（如 `configs/ft_promoter_cls.yaml` 和 `configs/ch_promoter_full_ddp.yaml`）也全部采用此 384/1536 配置，这正是原生 ChromBERT 的真实参数。另外，原生论文中采用 Post-LN 范式，而在 FlashChromBert 现代复现中升级为 Pre-LN。

### 5.2 序列切分配置 (Tokenization)
*   **K-mer**: $k = 4$
*   **Vocabulary Size**: 15态下词表大小为 50,630（包含特殊 token）。
*   **滑动步长 (Stride)**: 预训练阶段采用 Stride 1；下游任务可灵活调整为 Stride 2 或 3，使得 512 token 的序列容量可覆盖极长的染色质上下文。

### 5.3 具体任务设定
*   **预训练 (Pretraining)**:
    *   **输入窗口**: TSS 上游 2 kb 到下游 4 kb。
    *   **掩码比例 (Mask Ratio)**: 15%。
    *   **最大学习率**: 4e-4。
*   **基因表达二分类微调 (Binary Classification)**:
    *   **目标**: 区分高表达（RPKM > 5）与低/不表达（RPKM = 0）。
    *   **指标与特征**: 平均 AUC 达到 0.90 以上。通过使用 Stride 3（覆盖最大 190 kb 上游与 100 kb 下游），可获得最优分类表现。
*   **基因表达回归微调 (Regression)**:
    *   **目标**: 直接预测连续的 Log-transformed RPKM 表达量。
    *   **指标与特征**: 最高 Pearson 相关系数 $r = 0.791$。最佳性能是在采用最长的上下文配置下（上游 100 kb，下游 90 kb）取得的，证明了长距离染色质状态间的生物学调控依赖。