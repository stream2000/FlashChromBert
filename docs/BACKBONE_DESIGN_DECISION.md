# Backbone 设计决策：为什么自写 minimal BERT，不用 HuggingFace

**日期**：2026-04-18
**决策**：FlashChromBERT 的 Transformer backbone 由自写的 ~360 行 minimal BERT 实现，**不**继承或包装 HuggingFace `transformers` 的 `BertModel`。

本文件记录做出该决策时的讨论过程，便于未来回顾背景、判断决策是否仍然成立。

---

## 一、问题背景

在新项目 FlashChromBERT 立项时，面临一个基础架构选择：

- **选项 A**：直接用 HF `transformers.BertModel`（或 `MegatronBertModel`、`ModernBERT` 等变体），通过继承/包装满足 ChromBERT 的特殊需求。
- **选项 B**：自写一个 minimal BERT（仅保留 encoder-only + MLM + 自定义长序列头 + 自定义注意力返回路径），约 360 行。

HF 路径看似"站在巨人肩膀上"，但需要深入讨论后才能判断是否真正划算。

---

## 二、三个层面的对比分析

### 2.1 代码实现层面

| 维度 | HF `BertModel` + 改造 | 自写 minimal BERT (~360 行) |
|------|----------------------|---------------------------|
| 代码体量 | 基类自带 1500+ 行（NSP、TF ckpt loading、`token_type_ids`、`past_key_values`、cross-attention…） | 只保留实际用到的路径，每一行都有目的 |
| 修改方式 | 子类 override + monkey-patch，跨版本脆弱（HF 4.36 改过 attention dispatch 接口，4.41 又改过一次） | 接口自己定义，forward 签名自己说了算 |
| Pre-LN 需求 | `MegatronBertModel` 是 Pre-LN 但绑定了 Megatron 的 embedding/init 约定；`ModernBERT` 用 RoPE，不是 drop-in | 在 `encoder.py` 里 LayerNorm 放置位置决定，零成本 |
| Motif 路径的 eager 切换 | 要重新 `from_pretrained(..., attn_implementation="eager")` 整体重载；推理期可行，但和训练态割裂 | 同一个 `nn.Module`，`forward(..., return_attn=True)` 走 eager 分支，权重共享 |
| 自定义 MLM masking（k-mer `MASK_LIST` 连续遮 k 个 token） | 要 override `BertForMaskedLM` 或在 DataCollator 侧硬塞，还要维护 HF 的 `MaskedLMOutput` dataclass 契约 | LightningModule 里直接写，不经过 HF output dataclass |

### 2.2 性能层面

- **热路径等价**：HF 现在也走 `F.scaled_dot_product_attention`（4.36+ 的 `sdpa` 后端），在 Ada 上自动选 FA2 kernel。**自写和 HF-SDPA 在 kernel 层是同一条路径**，不存在"自写更快"。
- **真正的差别**：
  - HF forward 里有大量 `if self.config.xxx` 分支（attn_mask 兼容性、head_mask、encoder_hidden_states 等），eager 下这些 Python 分支有非零开销，`torch.compile` 时更易触发 graph break。自写版本路径干净，`torch.compile` 更稳。
  - `flash-attn` **包**（不是 SDPA）和 `output_attentions=True` 互斥是设计决定（FA2 不产出注意力矩阵）。用 HF 要么放弃 FA2 走 eager、要么放弃 motif 提取；自写用 `return_attn` flag，训练 FA2 与分析 eager 用**同一份权重**，编译期开关。
  - `flash-attn` 包需要和 torch/CUDA 版本严格匹配（编译时长可超 30 min，跨 driver 升级经常坏）。SDPA 路径只依赖 `torch ≥ 2.2`，零额外依赖。

### 2.3 上下游适配层面

- **上游（数据）**：tokenizer 是自定义 k-mer，不用 HF `tokenizers`；数据 pipeline 是 `css_utility.py` 输出，不走 HF `datasets`。HF `BertModel` 的输入契约（`token_type_ids`、特定 attention_mask 形状）反而是额外适配负担。自写可以让 forward 签名就是实际需要的几个张量。
- **下游（任务头）**：两个关键头（`BertForLongSequenceClassification` / `Cat`）本就是自定义，HF 现成类帮不上忙。Motif 分析层需要直读中间层注意力——自写的 encoder 把 per-layer attention 作为一等返回值，HF 需 `output_attentions=True` 触发 eager 全量再算一遍。
- **生态吸引力反向**：HF 的最大收益是 Hub 预训练权重、Trainer、PEFT。ChromBERT 的 k-mer 词表没有任何 HF 预训练 checkpoint 可直接用，且已决定 Lightning + 从头训——**生态收益为零**。

---

## 三、专题一：FA2 与 motif 提取的冲突本质

该冲突是驱动"自写"决策的最关键技术点之一，需要单独展开。

### 3.1 标准注意力 vs FA2 的关键区别

**标准（eager）注意力**是三步显式张量操作：

```
Q, K, V:  [B, H, L, D]
S = Q @ K^T / sqrt(D)    →  [B, H, L, L]   ← 注意力分数矩阵
P = softmax(S)            →  [B, H, L, L]   ← 注意力权重矩阵 ★
O = P @ V                 →  [B, H, L, D]
```

`P` 以完整 tensor 形式存在于显存，可以 `.detach().cpu()` 取出——这正是 `motif_utils.py` 的输入。

**FA2** 的核心创新就是**从来不生成完整的 `P`**：

```
FA2 kernel (CUDA 层面):
    将 Q, K, V 切成小块 (tile)
    for block_i in Q_blocks:
        for block_j in K_blocks, V_blocks:
            s_ij = q_i @ k_j^T / sqrt(D)        # 小块
            p_ij = online_softmax(s_ij, 累计量)  # 在线 softmax
            o_i += p_ij @ v_j                    # 直接累加到输出
            # ← p_ij 用完即丢，不写回 HBM
    写回 o
```

关键事实：
- `P` **从来没有在 GPU 内存中作为完整张量存在过**
- 按 tile 在寄存器/SRAM 内算出来、用完即丢
- FA2 kernel **没有"返回 P"的接口**，物理上就没这个东西
- 这正是 FA2 快的原因：省掉 O(L²) 的 HBM 读写（HBM 带宽是 attention 的真实瓶颈）

### 3.2 所以冲突是这样的

| 需求 | 所需路径 |
|------|---------|
| 训练速度（长序列尤其关键） | FA2 kernel → 不产出 P |
| Motif 提取（读注意力矩阵找高权重区域） | eager 路径 → 必须产出 P |

这**不是** HF 实现的问题——在任何框架里都冲突，因为是底层 kernel 决定的。HF 在 config 里遇到 `attn_implementation="flash_attention_2"` + `output_attentions=True` 会直接报错。

### 3.3 HF 的"解法"为什么难受

HF 官方解法：**推理时重新 `from_pretrained`，指定 `attn_implementation="eager"`**。

```python
# 训练
model = AutoModel.from_pretrained(ckpt, attn_implementation="flash_attention_2")
# 要提取 motif 时：
model_for_motif = AutoModel.from_pretrained(ckpt, attn_implementation="eager")
```

工程上的问题：
1. **两个模型对象**：显存要么存两份，要么存完再加载（IO 开销）
2. **权重本是同一份**：FA2 和 eager 只是 kernel 差异，没必要走完整 load 流程
3. **训练/分析割裂**：训练中途想做 motif 快照（观察 attention 学成什么样）很别扭——save ckpt → reload eager → forward → 切回训练
4. **Attention mask / dtype 细节差异**：FA2 和 eager 对 padding 处理有细微差异，HF 里不总是完全对齐

### 3.4 自写 minimal BERT 的解法

把 flag 做成 attention 层入参，权重共享、代码分支：

```python
class MultiHeadAttention(nn.Module):
    def forward(self, x, return_attn: bool = False):
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1)
        # ... reshape to [B, H, L, D]
        if return_attn:
            # eager path: 显式算 P 返回
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            p = F.softmax(scores, dim=-1)
            o = p @ v
            return o, p              # ← motif_utils 要的 P
        else:
            # 快路径: SDPA 自动选 FA2 kernel
            o = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            return o, None
```

同一个 `nn.Module` 实例、同一份权重、训练时 `return_attn=False`、motif 分析时 `return_attn=True`。零重载，零割裂。

---

## 四、专题二：HF adapter 推迟到下游做，为什么更划算

这是决定"要不要为未来的迁移学习提前绑 HF"的关键判断。

### 4.1 两种策略的成本账

**策略 A（一开始就绑 HF backbone）**：

整个 codebase 从第一天起遵守 HF 契约：
- Forward 签名：`input_ids, attention_mask, token_type_ids, position_ids, head_mask, ...` 一串都要接
- 输出要包成 `BaseModelOutputWithPooling` dataclass
- 配置继承 `PretrainedConfig`，用 `from_pretrained` / `save_pretrained`
- Tokenizer 挂到 `PreTrainedTokenizer` 接口（k-mer tokenizer 原本 ~50 行，挂上去变 ~300 行）
- LightningModule 到处是 `outputs.last_hidden_state` / `outputs.attentions` 这种包装访问

HF 每 6-9 个月一次破坏性变更时，整条链都要改。而这一切是为了**保留一个还没决定要不要用的可能性**。

**策略 B（自写 backbone + 需要时再写 adapter）**：

Backbone 自己的——forward 签名就几个张量、config 是 `@dataclass`、权重就是普通 `state_dict`。

真要接 DNABERT-2 做迁移学习那天，写一个独立的 adapter 文件：

```python
# src/chrombert/integrations/hf_adapter.py
# 只在想做迁移学习时被 import，不影响主流程
from transformers import AutoModel
from chrombert.model import ChromBERTModel

# HF 参数名 → 自己的参数名 的映射表
HF_TO_CHROMBERT = {
    "encoder.layer.{i}.attention.self.query.weight":  "encoder.blocks.{i}.attn.q_proj.weight",
    "encoder.layer.{i}.attention.self.key.weight":    "encoder.blocks.{i}.attn.k_proj.weight",
    "encoder.layer.{i}.attention.self.value.weight":  "encoder.blocks.{i}.attn.v_proj.weight",
    "encoder.layer.{i}.attention.output.dense.weight":"encoder.blocks.{i}.attn.out_proj.weight",
    # ... 约 20 行
}

def load_dnabert2_into_chrombert(hf_ckpt: str, model: ChromBERTModel, *,
                                  load_embeddings: bool = False) -> list[str]:
    """Load HF DNABERT-2 weights into custom ChromBERT.
    Returns list of parameters NOT loaded (for inspection)."""
    hf_model = AutoModel.from_pretrained(hf_ckpt)
    hf_state = hf_model.state_dict()
    new_state = {}
    for hf_name, hf_tensor in hf_state.items():
        our_name = remap(hf_name, HF_TO_CHROMBERT)
        if our_name is None: continue
        if not load_embeddings and "embeddings" in our_name: continue
        new_state[our_name] = hf_tensor
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    return list(missing) + list(unexpected)
```

整个 adapter ~100 行，一个文件，**不触碰 backbone 任何地方**。不行就删掉，主代码零影响。

### 4.2 对 ChromBERT 尤其合理

盘一下 HF Hub 上 DNA 模型，迁移学习对 ChromBERT 的真实价值**很薄**：

| 模型 | Vocab | 架构 | 能迁移到 ChromBERT 吗？ |
|------|-------|------|----------------------|
| DNABERT (v1) | 3-6 mer 核苷酸 | 标准 BERT | 词表不同（ChromBERT 是 CSS 染色质状态 k-mer，不是 ACGT），**embedding 不能迁移** |
| DNABERT-2 | BPE of nucleotides | BERT + ALiBi | 词表不同 + ALiBi 位置编码（ChromBERT 用绝对位置）→ **embedding + position 都不能迁移** |
| Nucleotide Transformer | 6-mer 核苷酸 | ESM-2 风格 + RoPE | 架构和 BERT 差很多，**只能迁移 FFN/attn 权重，且位置编码整个换掉** |

能迁移的只有 **transformer block 的 attn + FFN 权重**——而这些权重是在"核苷酸序列的语言模型任务"上学的，和 CSS（染色质状态序列）任务**在语义层面关系很弱**。

结论：
- **迁移学习的收益本身存疑**（不像 NLP 里 BERT→下游 NLP 那种强收益）
- 值得投入的工作量应该是"需要时再写的 100 行 adapter"
- 不是"从头绑 HF 以便将来好接"

### 4.3 一般性原则

这是 **speculative generality（投机性通用化）** 反模式的典型案例：

> 为未来可能用到的功能，现在付出架构成本。

正确做法：
- 主流程保持**最小、干净、任务专用**
- 集成第三方生态的代码**隔离在 `integrations/` 或 `adapters/` 子模块**
- 这些子模块**按需编写**、**独立演化**、**删掉不影响主流程**

与 legacy 迁移策略一致（`src/<pkg>/legacy/` copy-verbatim，adapter 放 sibling 模块）——**把生态耦合关在盒子里**。

---

## 五、决策总结

对 FlashChromBERT 这个特定项目，**自写 minimal BERT 是正确选择**，理由不是"自写更快"，而是：

1. **HF 最值钱的东西（Hub 权重、Trainer、tokenizer）一个都用不上**
2. **HF 最碍事的地方（FA2 与注意力矩阵互斥）正好卡在 motif 提取的必经之路上**
3. **360 行自写代码的维护成本，低于跨 HF 版本做 override/monkey-patch 的长期成本**

### 何时应反悔去用 HF

如果未来想接 HF Hub 上的 DNA 预训练模型做迁移学习，那时写一个 `integrations/hf_adapter.py` 即可——而不是反过来把主 backbone 绑上 HF。
