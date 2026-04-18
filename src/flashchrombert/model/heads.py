import torch
import torch.nn as nn

from .config import BertConfig


class MLMHead(nn.Module):
    """Standard BERT MLM head. Output projection ties to token embedding weights."""

    def __init__(self, config: BertConfig):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        )
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def tie_weights(self, token_embedding: nn.Embedding) -> None:
        self.decoder.weight = token_embedding.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.transform(hidden_states))
