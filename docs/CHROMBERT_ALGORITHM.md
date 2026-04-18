# ChromBERT 关键预处理与预训练算法摘要

本文档基于 ChromBERT 最新预印本论文（reference/chrombert-latest.pdf）提取，总结了其核心的预处理和预训练算法流程，作为本项目的参考事实。

## 1. 数据预处理 (Data Preprocessing)

ChromBERT 将原始的表观基因组信号转换为可供自然语言处理模型（NLP）学习的离散序列：

*   **数据来源与定义**：使用 ROADMAP 项目提供的 127 种人类细胞和组织类型的染色质状态注释（基于 hg19，通过 ChromHMM 以 200-bp 为区间进行标注）。主要使用的是 15-state 模型（15 种不同的染色质状态）。
*   **字母表映射 (Alphabetical Encoding)**：为了适应 NLP 模型，将以数字 (1-15) 表示的染色质状态直接映射为相应的英文字母 (A - O)。每一个字母代表一个 200-bp 的染色质区段状态。
*   **端粒过滤**：为了消除染色体端粒（telomeres）可能引入的偏差，预处理时剔除了每条染色体两端各 10,000 bp 的序列。
*   **K-mer 词表化 (Tokenization)**：
    *   采用了类似于 DNABERT 的 **k-mer 连续重叠切分**策略。
    *   由于 15 种状态构成的词表规模会随 $k$ 的增加而急剧膨胀（5-mer 和 6-mer 词表极大，导致 GPU 显存受限），最终**选择 4-mers** 作为预训练和微调的主要 Token 单元，词表大小为 50,630（包含特殊 token）。
    *   在特定任务（如长序列输入）中，也探索了步长（stride）大于 1 的 tokenization 策略（如 stride=2 或 _3），以便捕获更广阔的基因组上下文。序列最大长度限制为 512 个 token。
    *   在采用 18 状态（IHEC 数据集，A-R）时，词表大小为 104,980（排除了全是 Quiescent 状态的 `RRRR` token）。

## 2. 预训练算法 (Pretraining)

预训练使得 ChromBERT 能够学习染色质状态在基因组中的一般分布规律和语法结构。

*   **模型架构**：基于标准的 BERT-base 架构（Transformer 编码器），包含 12 层 Transformer layers，12 个注意力头（attention heads），隐藏层维度为 384 或 768（论文中对参数的描述在基础框架和具体应用段落略有区分，基础设为 hidden size 768, intermediate layer 3072，微调阶段部分配置使用了 hidden size 384, intermediate layer 1536），使用 GELU 激活函数。
*   **预训练目标：掩码语言模型 (Masked Language Modeling, MLM)**：
    *   遵循标准 BERT 的 MLM 目标，随机掩盖（mask）输入序列中 **15%** 的 tokens。
    *   训练模型根据上下文预测被掩盖的 k-mer tokens。
*   **训练策略**：
    *   分块学习：为了处理全基因组的海量数据，采用分块（Chunks）顺序训练策略，利用之前块的模型权重初始化下一块，逐步优化全基因组的模型权重。
    *   区域聚焦：除了全基因组预训练外，论文还特别进行了**启动子区域 (Promoter-specific)**（TSS 上游 2kb 至下游 4kb）的专项预训练。
    *   评估指标：困惑度 (Perplexity) 随着训练稳步下降至 1.0 左右，表明模型能有效捕捉染色质状态极其稳定的局部序列结构。_

## 3. 下游分析：Motif 提取与聚类算法

ChromBERT 不仅仅输出预测特征，还利用注意力机制提取“染色质状态 Motif”：

*   **高注意力区域提取**：在二分类或回归微调后，利用输出注意力矩阵（Attention matrix），提取注意力分数显著高于平均阈值且高频出现的序列区域，通过 FDR (False Discovery Rate) 过滤后定义为候选 Motif。
*   **动态时间规整 (Dynamic Time Warping, DTW)**：由于染色质状态序列（甚至具有相同生物学功能的区域）长度高度可变且动态，普通的序列比对不适用。引入了语音识别领域的 DTW 算法，通过最小化累积距离来计算不同长度、不同节奏的染色质 Motif 之间的结构相似度（兼容正向-正向和正向-反向的相似度比对）。
*   **凝聚层次聚类 (Agglomerative Clustering)**：基于 DTW 计算的相似度分数，将这些不同长度的 Motif 聚类为具有代表性的功能模式类别（并配合 UMAP 进行降维可视化）。
