"""LightningModule for promoter fine-tuning (classification + regression).

Wraps `BertForSequenceClassification`, loads a pretrained Lightning MLM
checkpoint into the backbone (with 'model.bert.' → 'bert.' prefix stripping),
and logs paper-style metrics:

- classification: accuracy, F1, AUC (val only)
- regression:    Pearson r, Spearman r, MSE (val only)
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..model import BertConfig, BertForSequenceClassification

TaskMode = Literal["classification", "regression"]


def _linear_warmup_decay(current: int, warmup: int, total: int) -> float:
    if current < warmup:
        return current / max(1, warmup)
    return max(0.0, (total - current) / max(1, total - warmup))


def load_pretrained_backbone(
    model: BertForSequenceClassification, ckpt_path: str | Path
) -> tuple[list[str], list[str]]:
    """Load `bert.*` weights from a Lightning MLM checkpoint (`LitBertMLM`).

    Legacy MLM key layout: `model.bert.<...>` / `model.mlm_head.<...>`.
    We keep the `bert.*` block and drop `mlm_head.*`.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    state: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if k.startswith("model.bert."):
            state[k[len("model."):]] = v
    return model.load_state_dict(state, strict=False)


class LitBertFinetune(L.LightningModule):
    """Fine-tuning module for promoter classification / regression.

    Metrics are accumulated in Python lists during validation and computed
    once at epoch end, so we get paper-style AUC / Pearson on the full dev set.
    """

    def __init__(
        self,
        config: BertConfig,
        task: TaskMode,
        num_labels: int,
        pretrained_ckpt: str | Path | None = None,
        classifier_dropout: float | None = None,
        freeze_backbone: bool = False,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        total_steps: int = 10_000,
        beta1: float = 0.9,
        beta2: float = 0.999,
        adam_eps: float = 1e-8,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])
        self.config = config
        self.task = task
        self.num_labels = num_labels

        self.model = BertForSequenceClassification(
            config, num_labels=num_labels, classifier_dropout=classifier_dropout
        )

        if pretrained_ckpt is not None:
            missing, unexpected = load_pretrained_backbone(self.model, pretrained_ckpt)
            # Expect: backbone loaded cleanly; head weights are missing
            # (random-init) and MLM decoder weights are absent from `state`
            # (so 'unexpected' is empty).
            print(
                f"[finetune] loaded {pretrained_ckpt}: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

        if freeze_backbone:
            for p in self.model.bert.parameters():
                p.requires_grad_(False)
            n_frozen = sum(p.numel() for p in self.model.bert.parameters())
            n_head = sum(p.numel() for p in self.model.head.parameters())
            print(f"[finetune] backbone frozen ({n_frozen:,} params); "
                  f"head trainable ({n_head:,} params)")

        self._val_preds: list[torch.Tensor] = []
        self._val_labels: list[torch.Tensor] = []

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def training_step(self, batch, batch_idx):
        out = self.model(**batch)
        self.log("train_loss", out.loss, prog_bar=True, on_step=True, on_epoch=True)
        return out.loss

    def validation_step(self, batch, batch_idx):
        out = self.model(**batch)
        self.log("val_loss", out.loss, prog_bar=True, sync_dist=True)
        # Cast to float32 on CPU to keep metric math numerically stable across
        # bf16 training precision.
        self._val_preds.append(out.logits.detach().float().cpu())
        self._val_labels.append(batch["labels"].detach().cpu())
        return out.loss

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return
        preds = torch.cat(self._val_preds, dim=0)
        labels = torch.cat(self._val_labels, dim=0)
        self._val_preds.clear()
        self._val_labels.clear()

        # Bug fix: gather preds/labels across DDP ranks before computing metrics.
        # Each rank accumulates only its own shard; averaging per-shard AUC/Pearson
        # is not equivalent to the global metric, so we must all_gather first.
        if self.trainer.world_size > 1:
            preds = self.all_gather(preds.to(self.device)).flatten(0, 1).cpu()
            labels = self.all_gather(labels.to(self.device)).flatten(0, 1).cpu()

        if self.task == "classification":
            # preds: [N, num_labels], labels: [N] long
            probs = torch.softmax(preds, dim=-1)
            pred_ids = probs.argmax(dim=-1)
            acc = (pred_ids == labels).float().mean().item()
            self.log("val_acc", acc, prog_bar=True, sync_dist=True)

            if self.num_labels == 2:
                from sklearn.metrics import f1_score, roc_auc_score

                y_true = labels.numpy()
                y_pred = pred_ids.numpy()
                y_score = probs[:, 1].numpy()
                f1 = f1_score(y_true, y_pred, zero_division=0)
                # roc_auc_score requires both classes present in y_true
                try:
                    auc = roc_auc_score(y_true, y_score)
                except ValueError:
                    auc = float("nan")
                self.log("val_f1", f1, prog_bar=True, sync_dist=True)
                self.log("val_auc", auc, prog_bar=True, sync_dist=True)
        else:
            # preds: [N, 1] or [N], labels: [N] float
            pred_vec = preds.view(-1)
            label_vec = labels.view(-1).float()
            mse = torch.mean((pred_vec - label_vec) ** 2).item()
            self.log("val_mse", mse, prog_bar=True, sync_dist=True)

            from scipy.stats import pearsonr, spearmanr

            pn = pred_vec.numpy()
            ln = label_vec.numpy()
            r_pearson = float(pearsonr(pn, ln).statistic)
            r_spearman = float(spearmanr(pn, ln).statistic)
            self.log("val_pearson", r_pearson, prog_bar=True, sync_dist=True)
            self.log("val_spearman", r_spearman, prog_bar=True, sync_dist=True)

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
        total = self.hparams.total_steps
        warmup = int(total * self.hparams.warmup_ratio)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda step: _linear_warmup_decay(step, warmup, total),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
