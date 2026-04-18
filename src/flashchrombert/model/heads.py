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


class SequenceClassificationHead(nn.Module):
    """CLS pooling (Linear + tanh) + dropout + linear classifier.

    Matches the legacy HuggingFace `BertPooler` + `BertForSequenceClassification`
    head shape, so behaviour is directly comparable with ChromBERT's promoter
    fine-tune runs. Used for both binary classification (num_labels=2) and
    scalar regression (num_labels=1).
    """

    def __init__(self, config: BertConfig, num_labels: int, dropout: float | None = None):
        super().__init__()
        self.num_labels = num_labels
        self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_act = nn.Tanh()
        self.dropout = nn.Dropout(
            config.hidden_dropout if dropout is None else dropout
        )
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls = hidden_states[:, 0]
        pooled = self.pooler_act(self.pooler_dense(cls))
        return self.classifier(self.dropout(pooled))
