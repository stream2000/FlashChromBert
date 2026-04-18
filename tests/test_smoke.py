"""End-to-end smoke test: a few training steps on the demo data must reduce loss."""

from pathlib import Path

import lightning as L
import torch

from flashchrombert.data import CharTokenizer, MLMDataModule
from flashchrombert.lightning import LitBertMLM
from flashchrombert.model import BertConfig


def test_cpu_smoke(tmp_path):
    torch.manual_seed(0)
    L.seed_everything(0)

    sample = Path(__file__).resolve().parents[1] / "data" / "sample.txt"
    assert sample.exists(), "demo sample.txt missing"

    tokenizer = CharTokenizer()
    cfg = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_position_embeddings=128,
    )

    dm = MLMDataModule(
        train_file=sample,
        val_file=None,
        tokenizer=tokenizer,
        batch_size=4,
        max_length=64,
        num_workers=0,
    )
    model = LitBertMLM(config=cfg, learning_rate=3e-3, warmup_steps=5, total_steps=50)

    trainer = L.Trainer(
        max_steps=30,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        enable_checkpointing=False,
        enable_progress_bar=False,
        logger=False,
        log_every_n_steps=5,
    )
    trainer.fit(model, dm)

    final = trainer.logged_metrics.get("train_loss_step") or trainer.logged_metrics.get(
        "train_loss"
    )
    assert final is not None
    # Random init CE ~ log(vocab_size). Must be below that after training.
    import math
    assert float(final) < math.log(tokenizer.vocab_size), (
        f"Loss did not decrease below random baseline: {float(final)}"
    )
