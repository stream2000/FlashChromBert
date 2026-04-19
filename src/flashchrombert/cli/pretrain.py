from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint

from ..data import (
    CharTokenizer,
    CStateTokenizer,
    KmerCStateTokenizer,
    KmerMaskListMaskingStrategy,
    MaskingStrategy,
    MLMDataModule,
    RandomFixedLengthDataModule,
    StandardMaskingStrategy,
    StreamingMLMDataModule,
    Tokenizer,
)
from ..lightning import LitBertMLM
from ..model import BertConfig


def build_tokenizer(spec: dict) -> Tokenizer:
    name = spec["type"]
    if name == "char":
        return CharTokenizer()
    if name == "cstate":
        return CStateTokenizer(num_states=spec.get("num_states", 15))
    if name == "kmer_cstate":
        k = spec["k"]
        num_states = spec.get("num_states", 15)
        vocab_file = spec.get("vocab_file")
        if vocab_file:
            return KmerCStateTokenizer.from_vocab_file(vocab_file, k=k, num_states=num_states)
        return KmerCStateTokenizer(k=k, num_states=num_states)
    raise ValueError(f"Unknown tokenizer: {name}")


def build_masking(spec: dict | None) -> MaskingStrategy:
    spec = spec or {"type": "standard"}
    name = spec.get("type", "standard")
    prob = spec.get("mlm_probability", 0.15)
    if name == "standard":
        return StandardMaskingStrategy(mlm_probability=prob)
    if name == "kmer_mask_list":
        return KmerMaskListMaskingStrategy(k=spec["k"], mlm_probability=prob)
    raise ValueError(f"Unknown masking: {name}")


def build_datamodule(cfg: dict, tokenizer: Tokenizer):
    data = cfg["data"]
    kind = data.get("kind", "file")
    masking = build_masking(cfg.get("masking"))
    if kind == "file":
        return MLMDataModule(
            train_file=data["train_file"],
            val_file=data.get("val_file"),
            tokenizer=tokenizer,
            batch_size=data["batch_size"],
            max_length=data["max_length"],
            num_workers=data.get("num_workers", 2),
            masking=masking,
        )
    if kind == "stream":
        return StreamingMLMDataModule(
            train_file=data["train_file"],
            val_file=data.get("val_file"),
            tokenizer=tokenizer,
            batch_size=data["batch_size"],
            max_length=data["max_length"],
            num_workers=data.get("num_workers", 2),
            masking=masking,
        )
    if kind == "random_fixed":
        return RandomFixedLengthDataModule(
            tokenizer=tokenizer,
            num_samples=data["num_samples"],
            seq_len=data["seq_len"],
            batch_size=data["batch_size"],
            num_workers=data.get("num_workers", 2),
            seed=cfg.get("seed", 0),
            masking=masking,
        )
    raise ValueError(f"Unknown data.kind: {kind}")


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser("fcbert-pretrain")
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    L.seed_everything(cfg.get("seed", 42))

    tokenizer = build_tokenizer(cfg["tokenizer"])

    model_cfg = BertConfig(vocab_size=tokenizer.vocab_size, **cfg["model"])

    dm = build_datamodule(cfg, tokenizer)

    lit_model = LitBertMLM(
        config=model_cfg,
        learning_rate=cfg["optimizer"]["learning_rate"],
        weight_decay=cfg["optimizer"].get("weight_decay", 0.01),
        warmup_steps=cfg["scheduler"].get("warmup_steps", 1000),
        total_steps=cfg["scheduler"]["total_steps"],
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg["trainer"].get("ckpt_dir", "checkpoints"),
            filename="{epoch}-{val_loss:.3f}",
            monitor="val_loss" if cfg["data"].get("val_file") else "train_loss",
            save_top_k=3,
            mode="min",
        )
    ]

    trainer = L.Trainer(
        max_epochs=cfg["trainer"].get("max_epochs", 10),
        max_steps=cfg["trainer"].get("max_steps", -1),
        precision=cfg["trainer"].get("precision", "bf16-mixed"),
        accelerator=cfg["trainer"].get("accelerator", "auto"),
        devices=cfg["trainer"].get("devices", "auto"),
        strategy=cfg["trainer"].get("strategy", "auto"),
        accumulate_grad_batches=cfg["trainer"].get("accumulate_grad_batches", 1),
        gradient_clip_val=cfg["trainer"].get("gradient_clip_val", 1.0),
        log_every_n_steps=cfg["trainer"].get("log_every_n_steps", 50),
        callbacks=callbacks,
    )

    trainer.fit(lit_model, dm)


if __name__ == "__main__":
    main()
