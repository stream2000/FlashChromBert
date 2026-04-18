# FlashChromBert 数据预处理与加载流水线 (Data Preprocessing Pipeline)

本文档详细说明了 FlashChromBert 在预训练阶段，数据从纯文本文件到最终输入进 BERT 模型前向传播的完整流转过程。整个过程高度模块化，由自定义的 Tokenizer、Dataset、DataLoader 以及特殊的连片掩码策略（Masking Strategy）协同完成。

## 1. 原始数据格式 (Raw Text)

预训练的数据输入是纯文本文件（例如 `train_split.txt` 或 `pretrain_genome_all.txt`）。
每一行代表一条染色质状态序列（CSS），内容为**空格分隔的 k-mer 字符串**。

*   **例子** (15种状态的 4-mer): `OOOO OOOO OOOI OOII OIII IIII...`
*   **来源**: 这些数据是由 `src/flashchrombert/legacy/css_utility.py` 中的脚本从原始 BED 文件生成并切分好的，FlashChromBert 直接消费这些文本文件以解耦重度依赖。

## 2. 分词与映射 (Tokenizer: `KmerCStateTokenizer`)

模型底层只进行张量运算，因此必须将文本 token 映射为整数 ID。这一步在 `src/flashchrombert/data/tokenizer.py` 中完成。

*   **词表构建 (Vocabulary)**: Tokenizer 初始化时会构建一个完整的词表。前 5 个为保留的特殊 Token (`[PAD], [UNK], [CLS], [SEP], [MASK]`)。随后是染色质状态字母（如 A-O 共 15 种）组合成的所有 4-mer（总计 $15^4 = 50625$ 种）。总词表大小为 50,630。
*   **切分 (Split)**: 由于原始文本已经是用空格分隔的，`_split` 方法仅调用 `text.split()` 进行极速切分，无需复杂的 BPE 或 WordPiece 算法。
*   **编码 (Encode)**: 将切分后的 4-mer 字符串映射成对应的整数 ID。默认会在句首自动插入 `[CLS]` (ID: 2)，句尾插入 `[SEP]` (ID: 3)。

## 3. 数据集读取 (Dataset: `MLMDataset`)

在 `src/flashchrombert/data/dataset.py` 中，`MLMDataset` 负责逐行读取和初步转换。

当 DataLoader 抓取一条样本时（`__getitem__`）：
1.  按索引读取对应行的文本内容。
2.  调用 `tokenizer.encode` 获取 Token ID 列表。
3.  根据配置的 `max_length` (如 512 或 2048) 进行暴力截断。
4.  将列表转换为 1 维 PyTorch `LongTensor` 返回。

## 4. 组装与掩码 (Collate & Masking)

这是整个流水线中最核心、算力最密集的一步。当 DataLoader 收集齐一个 Batch 的样本后，会调用自定义的组装函数 `collate_mlm`。

### 4.1 动态 Padding 与 Attention Mask
*   `collate_mlm` 会找出当前 Batch 中最长的序列，将所有较短的序列尾部用 `[PAD]` (ID: 0) 对齐填补。
*   同时生成 `attention_mask`，真实存在的 Token 位置标记为 `1`，Padding 填充的位置标记为 `0`。
*   **性能优化**: 如果全 Batch 都没有 Padding（例如全是 512 满长度），`attention_mask` 会被直接设为 `None`。这允许底层的 PyTorch `F.scaled_dot_product_attention` 自动分发到极速的 **FlashAttention-2** 硬件内核。

### 4.2 k-mer 连片掩码策略 (`KmerMaskListMaskingStrategy`)
标准的随机 15% 掩码在 k-mer 序列上会失效，因为相邻的 k-mer 有大量字母重叠（如 "ABCD" 与 "BCDE"）。如果只遮挡一个 4-mer，模型通过偷看左右邻居的字母重叠就能轻易猜出答案（信息泄露）。

FlashChromBert 实现了严格的连片扩展遮罩：
1.  首先按 15% 概率选取掩码“中心点”。
2.  查表 `MASK_LIST[4] = [-1, 1, 2]`，将中心点左右连续的几个 token **强行绑定，连片遮盖**。
3.  **替换原则**: 对选中的连续区域，执行标准的 BERT 扰动：80% 替换为 `[MASK]` ID，10% 随机替换为词表内其他 Token ID，10% 保持原样。
4.  **生成标签 (Labels)**: 将被选中的位置保留原本正确的 ID 作为监督目标，其他无需预测的非掩码位置全部标记为 `-100`。PyTorch 的 `CrossEntropyLoss` 会自动忽略 `-100` 从而只计算掩码位置的损失。

## 5. 喂给模型 (Forward Pass)

经过流水线的层层处理，最终由 DataLoader 输出给 Lightning 引擎的 `batch` 是一个干净的字典结构：

```python
{
    "input_ids": tensor([[2, 45, 4, 87, ...]]),       # 包含 [MASK] ID(4) 和扰动后的输入
    "attention_mask": tensor([[1, 1, 1, 1, ...]]),  # 标定序列有效长度
    "labels": tensor([[-100, -100, 87, -100, ...]])   # Ground Truth, -100被忽略
}
```

在 `src/flashchrombert/lightning/mlm.py` 中，这个字典被直接解包并输入到自定义的极简 BERT 中：

```python
def training_step(self, batch, batch_idx):
    # 前向传播，模型内部会提取 hidden_states 并通过 MLMHead 输出分类 Logits，
    # 随后直接与 labels 计算 CrossEntropyLoss。
    out = self.model(**batch) 
    return out.loss
```

这种模块化设计将繁重的 I/O 和复杂的张量逻辑全部下推到了 CPU 上的 DataLoader Worker 进程中，确保 GPU 始终只需执行最纯粹的矩阵乘法，从而实现了极高吞吐的预训练。
