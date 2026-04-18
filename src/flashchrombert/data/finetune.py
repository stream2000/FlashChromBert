"""TSV-based sequence-label dataset for promoter fine-tuning.

Mirrors the layout used by ChromBERT `DnaPromProcessor` /
`GeneExpressionProcessor`: one file per split (train.tsv, dev.tsv) with two
tab-separated columns — a whitespace-separated k-mer sequence and a label.

- Classification TSVs have a header row `sequence\tlabel`.
- Regression TSVs are headerless; the label is a continuous float (log-RPKM).

The dataset is task-agnostic; the caller selects int vs float label dtype.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Literal

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from .tokenizer import Tokenizer

TaskMode = Literal["classification", "regression"]


def _read_tsv(
    path: str | Path, has_header: bool, max_samples: int | None
) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if has_header and i == 0:
                continue
            line = line.rstrip("\n\r")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            rows.append((parts[0], parts[1]))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


class SeqLabelDataset(Dataset):
    """Load (sequence, label) pairs from a TSV. Tokenization is lazy per item."""

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: Tokenizer,
        task: TaskMode,
        max_length: int = 512,
        has_header: bool | None = None,
        max_samples: int | None = None,
    ):
        self.tokenizer = tokenizer
        self.task = task
        self.max_length = max_length
        if has_header is None:
            has_header = task == "classification"
        self.rows = _read_tsv(file_path, has_header=has_header, max_samples=max_samples)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float | int]:
        seq, label = self.rows[idx]
        ids = self.tokenizer.encode(seq, add_special=True)[: self.max_length]
        label_val: float | int
        if self.task == "classification":
            label_val = int(label)
        else:
            label_val = float(label)
        return torch.tensor(ids, dtype=torch.long), label_val


def collate_seq_label(
    batch: list[tuple[torch.Tensor, float | int]],
    pad_id: int,
    max_length: int,
    task: TaskMode,
) -> dict[str, torch.Tensor]:
    """Right-pad token ids. Emit `attention_mask=None` when the batch is full-length."""
    lengths = [min(len(x[0]), max_length) for x in batch]
    target_len = max(lengths)
    input_ids = torch.full((len(batch), target_len), pad_id, dtype=torch.long)
    attn = torch.zeros((len(batch), target_len), dtype=torch.long)
    for i, (seq, _) in enumerate(batch):
        n = lengths[i]
        input_ids[i, :n] = seq[:n]
        attn[i, :n] = 1

    if task == "classification":
        labels = torch.tensor([y for _, y in batch], dtype=torch.long)
    else:
        labels = torch.tensor([y for _, y in batch], dtype=torch.float32)

    no_padding = all(n == target_len for n in lengths)
    return {
        "input_ids": input_ids,
        "attention_mask": None if no_padding else attn,
        "labels": labels,
    }


class SeqLabelDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_file: str | Path,
        val_file: str | Path | None,
        tokenizer: Tokenizer,
        task: TaskMode,
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 2,
        has_header: bool | None = None,
        max_train_samples: int | None = None,
        max_val_samples: int | None = None,
    ):
        super().__init__()
        self.train_file = train_file
        self.val_file = val_file
        self.tokenizer = tokenizer
        self.task = task
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.has_header = has_header
        self.max_train_samples = max_train_samples
        self.max_val_samples = max_val_samples

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = SeqLabelDataset(
            self.train_file,
            self.tokenizer,
            self.task,
            max_length=self.max_length,
            has_header=self.has_header,
            max_samples=self.max_train_samples,
        )
        self.val_ds = (
            SeqLabelDataset(
                self.val_file,
                self.tokenizer,
                self.task,
                max_length=self.max_length,
                has_header=self.has_header,
                max_samples=self.max_val_samples,
            )
            if self.val_file
            else None
        )

    def _collate(self):
        return partial(
            collate_seq_label,
            pad_id=self.tokenizer.pad_token_id,
            max_length=self.max_length,
            task=self.task,
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
            return []
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate(),
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
