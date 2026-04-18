from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import Tokenizer


class MaskingStrategy(ABC):
    """Pluggable MLM masking. Subclass for k-mer aware masking (MASK_LIST scheme)."""

    @abstractmethod
    def mask(
        self, input_ids: torch.Tensor, tokenizer: Tokenizer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (inputs, labels) with labels=-100 at non-masked positions."""


# Contiguous-k-mer mask expansion — copied verbatim from the legacy
# ChromBERT run_pretrain.py (MASK_LIST + mask_tokens). Kept here so the
# preprocessing pipeline (chrombert_utils) stays untouched upstream.
MASK_LIST: dict[int, list[int]] = {
    3: [-1, 1],
    4: [-1, 1, 2],
    5: [-2, -1, 1, 2],
    6: [-2, -1, 1, 2, 3],
}


class KmerMaskListMaskingStrategy(MaskingStrategy):
    """Legacy ChromBERT MLM masking: pick 15% centers, then expand each with
    `MASK_LIST[k]` so that overlapping k-mers get masked together. Finally
    apply the standard 80/10/10 replacement scheme.

    Behavior mirrors `mask_tokens` in ChromBERT/training/examples/run_pretrain.py.
    """

    def __init__(self, k: int, mlm_probability: float = 0.15):
        if k not in MASK_LIST:
            raise ValueError(f"No MASK_LIST entry for k={k}; supported: {sorted(MASK_LIST)}")
        self.k = k
        self.mlm_probability = mlm_probability

    def mask(
        self, input_ids: torch.Tensor, tokenizer: Tokenizer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = input_ids.clone()
        labels = input_ids.clone()

        mask_list = MASK_LIST[self.k]
        prob = torch.full(labels.shape, self.mlm_probability)
        specials = tokenizer.special_token_ids()
        special_mask = torch.zeros_like(labels, dtype=torch.bool)
        for sid in specials:
            special_mask |= labels == sid
        prob.masked_fill_(special_mask, 0.0)
        pad_mask = labels == tokenizer.pad_token_id
        prob.masked_fill_(pad_mask, 0.0)

        masked_indices = torch.bernoulli(prob).bool()

        # Expand each selected center with MASK_LIST offsets, clamped to
        # [1, end], where `end` is the last position with non-zero probability.
        for i in range(masked_indices.shape[0]):
            nonzero = torch.nonzero(prob[i] != 0, as_tuple=False)
            if nonzero.numel() == 0:
                continue
            end = int(nonzero[-1].item())
            centers = set(torch.nonzero(masked_indices[i], as_tuple=False).view(-1).tolist())
            new_centers = set(centers)
            for c in centers:
                for off in mask_list:
                    j = c + off
                    if 1 <= j <= end:
                        new_centers.add(j)
            if new_centers:
                idx = torch.tensor(sorted(new_centers), dtype=torch.long)
                masked_indices[i, idx] = True

        labels[~masked_indices] = -100

        replace_mask = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[replace_mask] = tokenizer.mask_token_id

        random_mask = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~replace_mask
        )
        random_tokens = torch.randint(
            tokenizer.vocab_size, labels.shape, dtype=torch.long
        )
        inputs[random_mask] = random_tokens[random_mask]

        return inputs, labels


class StandardMaskingStrategy(MaskingStrategy):
    """Standard BERT: mask 15% of tokens, of which 80% → [MASK], 10% → random, 10% → unchanged."""

    def __init__(self, mlm_probability: float = 0.15):
        self.mlm_probability = mlm_probability

    def mask(
        self, input_ids: torch.Tensor, tokenizer: Tokenizer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inputs = input_ids.clone()
        labels = input_ids.clone()

        prob = torch.full(labels.shape, self.mlm_probability)
        specials = tokenizer.special_token_ids()
        special_mask = torch.zeros_like(labels, dtype=torch.bool)
        for sid in specials:
            special_mask |= labels == sid
        prob.masked_fill_(special_mask, 0.0)

        masked = torch.bernoulli(prob).bool()
        labels[~masked] = -100

        replace_mask = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked
        inputs[replace_mask] = tokenizer.mask_token_id

        random_mask = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked & ~replace_mask
        )
        random_tokens = torch.randint(tokenizer.vocab_size, labels.shape, dtype=torch.long)
        inputs[random_mask] = random_tokens[random_mask]

        return inputs, labels


class MLMDataset(Dataset):
    """Line-by-line MLM dataset. Each line is tokenized independently, truncated, padded."""

    def __init__(
        self,
        file_path: str | Path,
        tokenizer: Tokenizer,
        max_length: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(file_path, encoding="utf-8") as f:
            self.lines = [line for line in (l.strip() for l in f) if line]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> torch.Tensor:
        ids = self.tokenizer.encode(self.lines[idx], add_special=True)
        ids = ids[: self.max_length]
        return torch.tensor(ids, dtype=torch.long)


class RandomFixedLengthDataset(Dataset):
    """Synthetic fixed-length token streams drawn uniformly from non-special ids.

    Every sample has exactly `seq_len` tokens, no CLS/SEP, no padding — so the
    collate emits no attention_mask and SDPA can select the FlashAttention kernel.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        num_samples: int,
        seq_len: int,
        seed: int = 0,
    ):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.seq_len = seq_len
        specials = tokenizer.special_token_ids()
        self.content_ids = torch.tensor(
            [i for i in range(tokenizer.vocab_size) if i not in specials],
            dtype=torch.long,
        )
        self.generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(hash((idx,)) & 0xFFFFFFFF)
        picks = torch.randint(
            high=len(self.content_ids), size=(self.seq_len,), generator=g
        )
        return self.content_ids[picks]


def collate_mlm(
    batch: list[torch.Tensor],
    tokenizer: Tokenizer,
    masking: MaskingStrategy,
    max_length: int,
) -> dict[str, torch.Tensor]:
    """Pad to the longest sequence in the batch (capped by max_length), apply masking."""
    pad_id = tokenizer.pad_token_id
    lengths = [min(len(x), max_length) for x in batch]
    target_len = max(lengths)

    input_ids = torch.full((len(batch), target_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), target_len), dtype=torch.long)
    for i, seq in enumerate(batch):
        n = lengths[i]
        input_ids[i, :n] = seq[:n]
        attention_mask[i, :n] = 1

    masked_inputs, labels = masking.mask(input_ids, tokenizer)
    # When every sample fills the batch (no padding), skip the mask so SDPA can
    # dispatch to FlashAttention on Ada+ (FA2 rejects non-null attn_mask).
    no_padding = all(n == target_len for n in lengths)
    return {
        "input_ids": masked_inputs,
        "attention_mask": None if no_padding else attention_mask,
        "labels": labels,
    }
