from __future__ import annotations

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..model import BertConfig, BertForMaskedLM


def _linear_warmup_decay(current: int, warmup: int, total: int) -> float:
    if current < warmup:
        return current / max(1, warmup)
    return max(0.0, (total - current) / max(1, total - warmup))


class LitBertMLM(L.LightningModule):
    def __init__(
        self,
        config: BertConfig,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        total_steps: int = 100_000,
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.model = BertForMaskedLM(config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        out = self.model(**batch)
        self.log("train_loss", out.loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_ppl", torch.exp(out.loss), prog_bar=False, on_step=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch)
        self.log("val_loss", out.loss, prog_bar=True, sync_dist=True)
        self.log("val_ppl", torch.exp(out.loss), prog_bar=True, sync_dist=True)
        return out.loss

    def configure_optimizers(self):
        no_decay = ("bias", "LayerNorm.weight", "norm.weight")
        decay_params, nodecay_params = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (nodecay_params if any(nd in name for nd in no_decay) else decay_params).append(p)

        optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": self.hparams.weight_decay},
                {"params": nodecay_params, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.adam_eps,
        )
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: _linear_warmup_decay(
                step, self.hparams.warmup_steps, self.hparams.total_steps
            ),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
