from __future__ import annotations

from functools import partial
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from .dataset import (
    MaskingStrategy,
    MLMDataset,
    RandomFixedLengthDataset,
    StandardMaskingStrategy,
    StreamingMLMDataset,
    collate_mlm,
)
from .tokenizer import Tokenizer


class MLMDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_file: str | Path,
        val_file: str | Path | None,
        tokenizer: Tokenizer,
        batch_size: int = 32,
        max_length: int = 128,
        num_workers: int = 2,
        masking: MaskingStrategy | None = None,
    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.masking = masking or StandardMaskingStrategy()

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = MLMDataset(self.train_file, self.tokenizer, self.max_length)
        self.val_ds = (
            MLMDataset(self.val_file, self.tokenizer, self.max_length)
            if self.val_file
            else None
        )

    def _collate(self):
        return partial(
            collate_mlm,
            tokenizer=self.tokenizer,
            masking=self.masking,
            max_length=self.max_length,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate(),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return []  # Lightning treats empty list as "no validation"
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate(),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class StreamingMLMDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_file: str | Path,
        val_file: str | Path | None,
        tokenizer: Tokenizer,
        batch_size: int = 32,
        max_length: int = 128,
        num_workers: int = 2,
        masking: MaskingStrategy | None = None,
    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.masking = masking or StandardMaskingStrategy()

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = StreamingMLMDataset(self.train_file, self.tokenizer, self.max_length)
        self.val_ds = (
            StreamingMLMDataset(self.val_file, self.tokenizer, self.max_length)
            if self.val_file
            else None
        )

    def _collate(self):
        return partial(
            collate_mlm,
            tokenizer=self.tokenizer,
            masking=self.masking,
            max_length=self.max_length,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate(),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        if self.val_ds is None:
            return []
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self._collate(),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


class RandomFixedLengthDataModule(L.LightningDataModule):
    """Synthetic fixed-length data — no padding, so SDPA can use FlashAttention.

    Intended for smoke-testing the mixed-precision + FA2 path on arbitrary
    tokenizers (chromatin state, k-mer, char). No validation split.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        num_samples: int,
        seq_len: int,
        batch_size: int = 32,
        num_workers: int = 2,
        masking: MaskingStrategy | None = None,
        seed: int = 0,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.masking = masking or StandardMaskingStrategy()
        self.seed = seed

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = RandomFixedLengthDataset(
            self.tokenizer, self.num_samples, self.seq_len, self.seed
        )

    def _collate(self):
        return partial(
            collate_mlm,
            tokenizer=self.tokenizer,
            masking=self.masking,
            max_length=self.seq_len,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self._collate(),
            pin_memory=True,
        )

    def val_dataloader(self):
        return []
