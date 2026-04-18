import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BertConfig


class MultiHeadAttention(nn.Module):
    """Self-attention with two execution paths:
    - Training / fast inference: F.scaled_dot_product_attention (auto-selects FA2 on Ada+).
    - `return_attn=True`: eager matmul path that also returns attention weights for visualization.
    """

    def __init__(self, config: BertConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.attn_dropout = config.attention_dropout

        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size)
        self.proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_dropout = nn.Dropout(config.hidden_dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, N, _ = hidden_states.shape
        qkv = self.qkv(hidden_states).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each (B, H, N, D)

        # attention_mask: (B, N) with 1 = keep, 0 = pad
        if attention_mask is not None:
            # (B, 1, 1, N) additive mask for the eager path; bool mask for SDPA
            bool_mask = attention_mask.bool()[:, None, None, :]
        else:
            bool_mask = None

        if return_attn:
            scale = self.head_dim**-0.5
            attn_scores = (q @ k.transpose(-2, -1)) * scale
            if bool_mask is not None:
                attn_scores = attn_scores.masked_fill(~bool_mask, float("-inf"))
            attn_probs = attn_scores.softmax(dim=-1)
            attn_probs_dropped = F.dropout(
                attn_probs, p=self.attn_dropout, training=self.training
            )
            context = attn_probs_dropped @ v
            attn_out = attn_probs
        else:
            context = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=bool_mask,
                dropout_p=self.attn_dropout if self.training else 0.0,
            )
            attn_out = None

        context = context.transpose(1, 2).reshape(B, N, -1)
        return self.out_dropout(self.proj(context)), attn_out
