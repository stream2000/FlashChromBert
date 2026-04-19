"""CLI entry for promoter fine-tuning.

Usage:
    fcbert-finetune --config configs/ft_promoter_cls.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from ..data import KmerCStateTokenizer, SeqLabelDataModule
from ..lightning import LitBertFinetune
from ..model import BertConfig
from .pretrain import build_tokenizer, load_config


def build_datamodule(cfg: dict, tokenizer) -> SeqLabelDataModule:
    data = cfg["data"]
    return SeqLabelDataModule(
        train_file=data["train_file"],
        val_file=data.get("val_file"),
        tokenizer=tokenizer,
        task=cfg["task"],
        batch_size=data["batch_size"],
        max_length=data.get("max_length", 512),
        num_workers=data.get("num_workers", 2),
        has_header=data.get("has_header"),
        max_train_samples=data.get("max_train_samples"),
        max_val_samples=data.get("max_val_samples"),
    )


def main(argv: list[str] | None = None) -> None:
    import torch
    torch.set_float32_matmul_precision("high")

    parser = argparse.ArgumentParser("fcbert-finetune")
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    L.seed_everything(cfg.get("seed", 42))

    task = cfg["task"]
    if task not in ("classification", "regression"):
        raise ValueError(f"task must be classification|regression, got {task}")

    tokenizer = build_tokenizer(cfg["tokenizer"])
    if not isinstance(tokenizer, KmerCStateTokenizer):
        print(f"[warn] non-kmer tokenizer: {type(tokenizer).__name__}")

    model_cfg = BertConfig(vocab_size=tokenizer.vocab_size, **cfg["model"])

    dm = build_datamodule(cfg, tokenizer)
    dm.setup()
    n_train = len(dm.train_ds)
    trainer_cfg = cfg["trainer"]
    batch_size = cfg["data"]["batch_size"]
    devices = trainer_cfg.get("devices", 1)
    n_devices = devices if isinstance(devices, int) else 1
    accum = trainer_cfg.get("accumulate_grad_batches", 1)
    steps_per_epoch = max(1, n_train // (batch_size * n_devices * accum))
    total_steps = steps_per_epoch * trainer_cfg.get("max_epochs", 10)
    print(
        f"[finetune] train={n_train} val={len(dm.val_ds) if dm.val_ds else 0} "
        f"steps/epoch≈{steps_per_epoch} total_steps={total_steps}"
    )

    num_labels = 2 if task == "classification" else 1
    lit = LitBertFinetune(
        config=model_cfg,
        task=task,
        num_labels=num_labels,
        pretrained_ckpt=cfg.get("pretrained_ckpt"),
        classifier_dropout=cfg.get("classifier_dropout"),
        freeze_backbone=cfg.get("freeze_backbone", False),
        learning_rate=cfg["optimizer"]["learning_rate"],
        weight_decay=cfg["optimizer"].get("weight_decay", 0.01),
        warmup_ratio=cfg["scheduler"].get("warmup_ratio", 0.1),
        total_steps=total_steps,
    )

    ckpt_dir = trainer_cfg.get("ckpt_dir", "checkpoints/finetune")
    monitor = trainer_cfg.get(
        "monitor",
        "val_auc" if task == "classification" else "val_pearson",
    )
    mode = trainer_cfg.get("monitor_mode", "max")
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="{epoch}-{" + monitor + ":.4f}",
            monitor=monitor,
            save_top_k=1,
            mode=mode,
        )
    ]

    logger = CSVLogger(
        save_dir=trainer_cfg.get("log_dir", "lightning_logs"),
        name=trainer_cfg.get("log_name", Path(args.config).stem),
    )

    trainer = L.Trainer(
        max_epochs=trainer_cfg.get("max_epochs", 10),
        max_steps=trainer_cfg.get("max_steps", -1),
        precision=trainer_cfg.get("precision", "bf16-mixed"),
        accelerator=trainer_cfg.get("accelerator", "auto"),
        devices=devices,
        strategy=trainer_cfg.get("strategy", "auto"),
        accumulate_grad_batches=accum,
        gradient_clip_val=trainer_cfg.get("gradient_clip_val", 1.0),
        log_every_n_steps=trainer_cfg.get("log_every_n_steps", 50),
        val_check_interval=trainer_cfg.get("val_check_interval", 1.0),
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(lit, dm)

    # Final report — best metrics from the checkpoint callback.
    best_score = callbacks[0].best_model_score
    best_path = callbacks[0].best_model_path
    report = {
        "config": args.config,
        "task": task,
        "monitor": monitor,
        "best_score": float(best_score) if best_score is not None else None,
        "best_ckpt": best_path,
    }
    out = trainer_cfg.get("report_file")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[finetune] wrote {out}")
    print("[finetune] best:", report)


if __name__ == "__main__":
    main()
