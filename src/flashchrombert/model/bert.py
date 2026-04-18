from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import BertConfig
from .embeddings import BertEmbeddings
from .encoder import BertEncoder
from .heads import MLMHead


@dataclass
class BertOutput:
    last_hidden_state: torch.Tensor
    attentions: list[torch.Tensor] | None = None


@dataclass
class MaskedLMOutput:
    loss: torch.Tensor | None
    logits: torch.Tensor
    attentions: list[torch.Tensor] | None = None


class BertModel(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0, std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0, std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> BertOutput:
        x = self.embeddings(input_ids)
        hidden, attns = self.encoder(x, attention_mask, return_attn)
        return BertOutput(last_hidden_state=hidden, attentions=attns)


class BertForMaskedLM(nn.Module):
    def __init__(self, config: BertConfig):
        super().__init__()
        self.config = config
        self.bert = BertModel(config)
        self.mlm_head = MLMHead(config)
        self.mlm_head.tie_weights(self.bert.embeddings.token)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_attn: bool = False,
    ) -> MaskedLMOutput:
        out = self.bert(input_ids, attention_mask, return_attn=return_attn)
        logits = self.mlm_head(out.last_hidden_state)

        loss = None
        if labels is not None:
            # labels == -100 at non-masked positions (PyTorch convention)
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        return MaskedLMOutput(loss=loss, logits=logits, attentions=out.attentions)
