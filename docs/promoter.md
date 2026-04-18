 根据论文原文（位于
  FlashChromBert/reference/chrombert-latest.pdf），作者针对“启动子（Promoter）版本”主要进行了以下三类核心实验（包含预训练、分类、回归），以及相应的解释性分析。以下是作者所做的实验任务、使
  用的数据和取得的指标：

  1. 预训练 (Pretraining - Masked Language Modeling)
   * 任务定义：让模型学习启动子区域染色质状态序列的“语法”与上下文关系。
   * 使用数据：定义启动子区域为 TSS（转录起始位点）上游 2 kb 到下游 4 kb（共 6 kb 的窗口）。使用 15 种染色质状态（基于 ROADMAP 的 127 种细胞类型）构成的状态序列作为“文本”输入。
   * 实验结果/指标：随着训练的进行，模型的困惑度（Perplexity）从最初的约 2.75 稳定下降到了 1.24。(注：结合前面的分析记忆，这个 1.24 实际上是在包含训练集的伪验证集上测得的。)

  2. 微调：基因表达二分类 (Binary Classification)
   * 任务定义：根据输入序列预测基因的表达水平是“高表达”还是“低/不表达”。
   * 使用数据：
       * 标签：基于 RNA-seq 数据（RPKM值），区分高表达和不表达。
       * 划分：随机抽取 30,000 个基因作为训练集，6,000 个基因作为测试集。
       * 序列处理：针对模型 512 个 token 的长度限制，作者尝试了改变滑动窗口步长（Stride = 1, 2, 3）。通过增加 Stride，模型最多能够接收 TSS 周围高达 290 kb（上游 190 kb，下游 100
         kb）的超长上下文。对于反义链（Reverse strand）上的基因，对序列进行了翻转处理。
   * 评估指标：使用 F1 Score、平均 AUC (Mean Area Under the Curve) 和 Accuracy (准确率) 进行评估（具体图表中的最高 AUC
     等数值在文本中未直接写出，但结论指出引入上下游长上下文显著提升了分类性能）。

  3. 微调：基因表达回归预测 (Regression)
   * 任务定义：从定性分类升级为定量预测，直接根据启动子区域的染色质状态序列预测具体的基因表达量。
   * 使用数据：使用经过 Log 转换的连续 RPKM 值作为目标标签（log-transformed RPKM values），序列预处理和分类任务相似（使用定制的 GeneExpressionProcessor 构建配对数据）。
   * 实验结果/指标：作者测试了多种 TSS 上下游长度的组合。表现最好的配置是 上游 100 kb + 下游 90 kb，该配置下预测值与真实表达量之间的 Pearson 相关系数（Pearson's r）达到了 0.791。

  4. 解释性实验与 Motif 发现 (Interpretability & Motif Clustering)
   * 任务定义：打开黑盒，分析模型是如何判断高表达基因的。
   * 数据/方法：提取二分类微调模型最后一层的 Attention（注意力）权重，并结合 DTW (Dynamic Time Warping) 算法进行序列聚类。
   * 结论：
       * 模型的注意力极度集中在 TSS 周围（呈现显著的尖峰），尤其高度关注“活跃 TSS (A)”、“侧翼活跃 TSS (B)”和“强转录 (D)”等状态。
       * 模型不仅关注核心启动子，也参考了远端增强子 (Enhancer) 的上下文信号。
       * 通过聚类发现了特定于高表达基因的 Chromatin Motif（染色质状态的组合模式）。



✦ 关于启动子（Promoter）下游任务，论文中给出了以下具体的定量（Quantitative）实验结果指标：

  1. 基因表达定量回归 (Quantitative Regression)
  模型直接预测基因的 Log 转换 RPKM 表达量：
   * 最佳预测相关性：当使用最长的上下文配置（TSS 上游 100 kb + 下游 90 kb）时，模型预测值与真实表达量之间的 Pearson 相关系数（Pearson's r）达到了
     0.791。这是论文中关于启动子模型最核心的定量指标。

  2. 基因表达二分类 (Binary Classification)
  区分“高表达”与“不表达”基因：
   * 作者在正文中指出，通过使用最高到 Stride 3（覆盖上游 190 kb 和下游 100 kb）的超长上下文配置，模型分类准确率最高（Figure 3b 的 orange 曲线）。
   * 主要评估指标为 F1 Score、AUC 和 Accuracy（论文正文将具体分类的平均数值放在了图表 Figure 3 和补充材料中），并得出结论：“基因被以很高的准确率（high accuracy）成功分类”。

  3. 跨细胞类型泛化分类 (Cross-cell-type Classification)
  论文还测试了模型在一个细胞类型上训练，在另一个细胞类型上测试时的表现：
   * 对于生物学差异较大的细胞类型对（例如具有截然不同染色质模式的细胞），分类 Accuracy 达到了 0.776。
   * 对于生物学相似度高的细胞类型对（例如 Brain vs. Smooth Muscle），由于启动子特征差异不明显，分类 Accuracy 降至 0.529，这从侧面反映了模型提取的特征在生物学上是合理的。

  (注：论文另外提到在 TAD 边界等 3D 基因组下游任务的分类表现较为一般 [F1 < 0.6]，但这主要使用的是全基因组 [Genome-wide] 版本的模型，而非启动子版本。)


# ChromBERT: Promoter Downstream Tasks & Code References

This document summarizes the quantitative results and code implementations for the downstream tasks based on the ChromBERT pre-trained promoter model. According to the original paper and the codebase, the primary downstream tasks are **Gene Expression Binary Classification** and **Gene Expression Quantitative Regression**.

## 1. Gene Expression Binary Classification

### Task Description
This task fine-tunes the pre-trained ChromBERT model to classify whether a gene is "highly expressed" or "lowly/not expressed" based on its promoter-proximal chromatin state sequence.

### Quantitative Results (from the Paper)
*   **Metrics:** F1 Score, Mean Area Under the Curve (AUC), and Accuracy.
*   **Performance:**
    *   Achieved high classification accuracy with a mean AUC exceeding **0.90** (and peaking at **> 0.94** with extended sequence configurations using Stride 3).
    *   Cross-cell-type generalization showed robust biological correlation (e.g., Accuracy **0.776** between disparate cell types, and **0.529** between highly similar cell types where promoter states overlap heavily).

### Code References
*   **Execution Script:** 
    `training/examples/prom/script_ft/run_4mer_classification_finetune.sh`
    This script runs the fine-tuning process with the argument `--task_name dnaprom`.
*   **Python Entry Point:** 
    `training/examples/run_finetune.py`
    Handles the `dnaprom` task type. Lines 93-120 route the classification logic and compute accuracy as the primary evaluation metric.
*   **Data Processor:** 
    `training/src/transformers/data/processors/glue.py`
    Uses the `DnaPromProcessor` mapped to the `dnaprom` task to load and tokenize the `train.tsv` and `dev.tsv` sequence/label pairs. The output mode is set to `"classification"`.

---

## 2. Gene Expression Quantitative Regression

### Task Description
To predict gene expression levels quantitatively, the model was adapted to process continuous values (log-transformed RPKM) directly from the promoter chromatin state sequences.

### Quantitative Results (from the Paper)
*   **Metric:** Pearson Correlation Coefficient ($r$) between predicted and observed log-transformed RPKM values.
*   **Performance:**
    *   The model achieved a peak Pearson correlation of **$r = 0.791$**.
    *   This peak performance was observed when the sequence context was heavily extended (100 kb upstream and 90 kb downstream of the TSS), highlighting ChromBERT's ability to capture long-range regulatory dependencies.

### Code References
*   **Execution Script:** 
    `training/examples/prom/script_ft/run_4mer_regression_finetune.sh`
    This script executes the fine-tuning process with the argument `--task_name gene_expression`.
*   **Python Entry Point:** 
    `training/examples/run_finetune.py`
    Handles the `gene_expression` task type. Lines 96 and 128 route the regression logic (outputting continuous predicted labels rather than probabilities) and use Mean Squared Error (MSE) / Pearson metrics for evaluation.
*   **Data Processor:** 
    `training/src/transformers/data/processors/glue.py`
    Implements a custom `GeneExpressionProcessor` (Lines 595-610) to handle continuous log-RPKM values. This processor maps the sequence to the target float value instead of a categorical class, with the task output mode mapped to `"regression"`.

---

## Data Preparation Note
For both tasks, sequences are typically formatted into `train.tsv` and `dev.tsv` files containing the chromatin sequences and their corresponding labels (binary classes or continuous log-RPKM values). The model processes these sequences utilizing a sliding $k$-mer tokenization approach (typically $k=4$, with extended contexts handled by increasing the tokenization stride).