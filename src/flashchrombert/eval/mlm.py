"""Shared MLM evaluation core.

Generates a deterministic, model-agnostic evaluation fixture (masked inputs,
labels, ground-truth ids, dataset unigram frequencies) that both the
FlashChromBert runner and the legacy HuggingFace runner consume to produce
directly comparable numbers.

Only depends on torch + flashchrombert.data. No transformers dep.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from ..data import KmerCStateTokenizer, KmerMaskListMaskingStrategy, Tokenizer


@dataclass
class MLMEvalFixture:
    """A frozen batch of masked sequences for evaluation.

    `input_ids` already has the 80/10/10 MLM corruption applied. `labels`
    is -100 outside masked positions. `original_ids` holds the pre-mask
    sequence (for copy-neighbor baseline and sanity checks).
    """

    dataset_name: str
    input_ids: torch.Tensor       # [N, T] long
    labels: torch.Tensor          # [N, T] long, -100 outside masked
    attention_mask: torch.Tensor  # [N, T] long (all-ones rows if no padding)
    original_ids: torch.Tensor    # [N, T] long, pre-mask
    unigram_counts: torch.Tensor  # [V] long — token counts over the dataset
    vocab_size: int
    pad_id: int
    special_ids: list[int]
    seed: int
    mlm_probability: float
    k: int


def iter_chunks(tokens: list[str], chunk_size: int) -> list[list[str]]:
    """Non-overlapping windows; drops trailing window shorter than 3 tokens."""
    chunks: list[list[str]] = []
    for start in range(0, len(tokens), chunk_size):
        window = tokens[start : start + chunk_size]
        if len(window) >= 3:
            chunks.append(window)
    return chunks


def _load_lines(
    path: str | Path,
    tokenizer: Tokenizer,
    max_content_len: int,
    max_samples: int | None,
) -> list[list[int]]:
    """Read file, split each line into non-overlapping windows, tokenize.

    Returns a list of variable-length token-id lists INCLUDING [CLS] / [SEP].
    """
    samples: list[list[int]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            toks = line.split()
            for window in iter_chunks(toks, max_content_len):
                ids = tokenizer.encode(" ".join(window), add_special=True)
                samples.append(ids)
                if max_samples is not None and len(samples) >= max_samples:
                    return samples
    return samples


def _collate(
    samples: list[list[int]],
    pad_id: int,
    target_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad to `target_len` (right-pad). Returns (input_ids, attention_mask)."""
    n = len(samples)
    input_ids = torch.full((n, target_len), pad_id, dtype=torch.long)
    attn = torch.zeros((n, target_len), dtype=torch.long)
    for i, ids in enumerate(samples):
        m = min(len(ids), target_len)
        input_ids[i, :m] = torch.tensor(ids[:m], dtype=torch.long)
        attn[i, :m] = 1
    return input_ids, attn


def build_fixture(
    data_path: str | Path,
    dataset_name: str,
    tokenizer: KmerCStateTokenizer,
    max_length: int = 512,
    max_samples: int | None = 10000,
    mlm_probability: float = 0.0375,
    seed: int = 20260418,
) -> MLMEvalFixture:
    """Build a deterministic masked fixture from a raw 4-mer text file."""
    k = tokenizer.k
    # Leave room for [CLS] + [SEP].
    samples = _load_lines(data_path, tokenizer, max_length - 2, max_samples)
    if not samples:
        raise ValueError(f"No samples found in {data_path}")

    target_len = min(max(len(s) for s in samples), max_length)
    original_ids, attn = _collate(samples, tokenizer.pad_token_id, target_len)

    # Unigram counts over content tokens (ignoring pad, specials).
    special_ids = sorted(tokenizer.special_token_ids())
    counts = torch.bincount(
        original_ids.view(-1), minlength=tokenizer.vocab_size
    ).clone()
    for sid in special_ids:
        counts[sid] = 0

    # Deterministic masking.
    g_prev = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        strat = KmerMaskListMaskingStrategy(k=k, mlm_probability=mlm_probability)
        input_ids, labels = strat.mask(original_ids, tokenizer)
    finally:
        torch.random.set_rng_state(g_prev)

    return MLMEvalFixture(
        dataset_name=dataset_name,
        input_ids=input_ids,
        labels=labels,
        attention_mask=attn,
        original_ids=original_ids,
        unigram_counts=counts,
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_token_id,
        special_ids=special_ids,
        seed=seed,
        mlm_probability=mlm_probability,
        k=k,
    )


def save_fixture(fix: MLMEvalFixture, path: str | Path) -> None:
    payload = {
        "dataset_name": fix.dataset_name,
        "input_ids": fix.input_ids,
        "labels": fix.labels,
        "attention_mask": fix.attention_mask,
        "original_ids": fix.original_ids,
        "unigram_counts": fix.unigram_counts,
        "vocab_size": fix.vocab_size,
        "pad_id": fix.pad_id,
        "special_ids": fix.special_ids,
        "seed": fix.seed,
        "mlm_probability": fix.mlm_probability,
        "k": fix.k,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_fixture(path: str | Path) -> MLMEvalFixture:
    p = torch.load(path, map_location="cpu", weights_only=False)
    return MLMEvalFixture(**p)


def _topk_acc(logits: torch.Tensor, labels: torch.Tensor, k: int) -> tuple[float, int]:
    """Accuracy over positions where labels != -100. Returns (acc, n_positions)."""
    valid = labels != -100
    if not valid.any():
        return 0.0, 0
    flat_logits = logits[valid]             # [M, V]
    flat_labels = labels[valid]             # [M]
    topk = flat_logits.topk(k, dim=-1).indices  # [M, k]
    correct = (topk == flat_labels.unsqueeze(-1)).any(dim=-1)
    return correct.float().mean().item(), int(valid.sum().item())


@torch.no_grad()
def compute_model_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """Compute loss / ppl / top1 / top5 from logits at masked positions.

    `logits`: [N, T, V]; `labels`: [N, T] with -100 at non-masked positions.
    Only use for small N — for large fixtures call `evaluate_model` instead.
    """
    V = logits.size(-1)
    loss = F.cross_entropy(
        logits.reshape(-1, V),
        labels.reshape(-1),
        ignore_index=-100,
    ).item()
    top1, n_pos = _topk_acc(logits, labels, 1)
    top5, _ = _topk_acc(logits, labels, 5)
    return {
        "loss": loss,
        "ppl": float(torch.exp(torch.tensor(loss)).item()),
        "top1": top1,
        "top5": top5,
        "n_masked": n_pos,
    }


@torch.no_grad()
def evaluate_model(
    fix: MLMEvalFixture,
    forward_fn,
    *,
    batch_size: int = 32,
    device: str = "cuda",
) -> dict:
    """Batch-wise forward + metric accumulation.

    `forward_fn(input_ids, attention_mask) -> logits [B, T, V]` — caller
    wraps the model (FCB or HF) so this function stays framework-agnostic.
    attention_mask may be None if the whole batch is unpadded.
    """
    N = fix.input_ids.size(0)
    total_loss_sum = 0.0
    total_n = 0
    top1_correct = 0
    top5_correct = 0

    for start in range(0, N, batch_size):
        stop = min(start + batch_size, N)
        inp = fix.input_ids[start:stop].to(device)
        lab = fix.labels[start:stop].to(device)
        attn = fix.attention_mask[start:stop].to(device)
        # If every row is fully-attended we can pass None (lets SDPA pick FA2).
        attn_in = None if (attn == 1).all().item() else attn

        logits = forward_fn(inp, attn_in)  # [B, T, V]
        V = logits.size(-1)

        # summed CE over valid positions
        valid = lab != -100
        m = int(valid.sum().item())
        if m == 0:
            continue
        flat_logits = logits[valid]            # [m, V]
        flat_labels = lab[valid]               # [m]
        loss_sum = F.cross_entropy(
            flat_logits, flat_labels, reduction="sum"
        ).item()
        top1 = flat_logits.argmax(dim=-1) == flat_labels
        top5 = flat_logits.topk(5, dim=-1).indices == flat_labels.unsqueeze(-1)
        top5_any = top5.any(dim=-1)

        total_loss_sum += loss_sum
        total_n += m
        top1_correct += int(top1.sum().item())
        top5_correct += int(top5_any.sum().item())

    loss = total_loss_sum / max(1, total_n)
    return {
        "loss": loss,
        "ppl": float(torch.exp(torch.tensor(loss)).item()),
        "top1": top1_correct / max(1, total_n),
        "top5": top5_correct / max(1, total_n),
        "n_masked": total_n,
    }


@torch.no_grad()
def compute_baselines(fix: MLMEvalFixture) -> dict[str, dict]:
    """Three predictor baselines evaluated on the fixture's masked positions.

    uniform     — uniform dist over non-special vocab
    unigram     — vocab-wide frequency from this dataset (ex-specials)
    copy_left   — copy the left neighbour token (right if left is special/pad)

    For each, we report loss/top1/top5. ppl = exp(loss).
    """
    V = fix.vocab_size
    labels = fix.labels
    valid = labels != -100
    n_pos = int(valid.sum().item())
    if n_pos == 0:
        return {}

    flat_labels = labels[valid]  # [M]
    out: dict[str, dict] = {}

    # ---------- uniform ----------
    specials = set(fix.special_ids)
    non_special_ids = torch.tensor(
        [i for i in range(V) if i not in specials], dtype=torch.long
    )
    uniform_p = torch.full((V,), 0.0)
    uniform_p[non_special_ids] = 1.0 / len(non_special_ids)
    eps = 1e-12
    uniform_logp = torch.log(uniform_p + eps)
    loss_uniform = -uniform_logp[flat_labels].mean().item()
    # top-k for a constant distribution: any k of the non-special ids work;
    # top-1 acc = 1/|V'|, top-5 acc = 5/|V'|
    n_vs = len(non_special_ids)
    out["uniform"] = {
        "loss": loss_uniform,
        "ppl": float(torch.exp(torch.tensor(loss_uniform)).item()),
        "top1": 1.0 / n_vs,
        "top5": min(5.0 / n_vs, 1.0),
        "n_masked": n_pos,
    }

    # ---------- unigram ----------
    counts = fix.unigram_counts.float()
    total = counts.sum().clamp_min(1.0)
    unigram_p = counts / total
    unigram_logp = torch.log(unigram_p + eps)
    loss_unigram = -unigram_logp[flat_labels].mean().item()
    # Constant predictor: top-k = k most frequent tokens.
    topk = unigram_p.topk(5).indices  # [5]
    top1 = (flat_labels == topk[0]).float().mean().item()
    top5 = torch.isin(flat_labels, topk).float().mean().item()
    out["unigram"] = {
        "loss": loss_unigram,
        "ppl": float(torch.exp(torch.tensor(loss_unigram)).item()),
        "top1": top1,
        "top5": top5,
        "n_masked": n_pos,
    }

    # ---------- copy_left ----------
    # Predict token at position t = original token at t-1 (skip specials/pad).
    orig = fix.original_ids
    pad_id = fix.pad_id
    N, T = orig.shape
    # Build a per-position "left valid token id" tensor.
    left_pred = torch.full_like(orig, fill_value=pad_id)
    for t in range(1, T):
        # default: value at t-1
        candidate = orig[:, t - 1]
        # if t-1 is special/pad, fall back to t-2 if any
        bad = torch.zeros_like(candidate, dtype=torch.bool)
        for sid in fix.special_ids:
            bad |= candidate == sid
        bad |= candidate == pad_id
        # Use right neighbour if left was bad and t+1 is in range & non-special
        if t + 1 < T:
            right = orig[:, t + 1]
            right_bad = torch.zeros_like(right, dtype=torch.bool)
            for sid in fix.special_ids:
                right_bad |= right == sid
            right_bad |= right == pad_id
            candidate = torch.where(bad & ~right_bad, right, candidate)
        left_pred[:, t] = candidate
    # As a constant-on-position prediction, treat the predicted id as argmax.
    flat_pred = left_pred[valid]
    top1 = (flat_pred == flat_labels).float().mean().item()
    # For loss, we give probability ~1 to the predicted token (smoothed)
    # Use a simple delta+uniform mixture to avoid inf loss:
    delta = 0.9
    smooth = (1.0 - delta) / V
    # loss = -log(delta + smooth) on hits, -log(smooth) on misses
    hits = (flat_pred == flat_labels).float()
    loss_copy = -(
        hits * torch.log(torch.tensor(delta + smooth))
        + (1 - hits) * torch.log(torch.tensor(smooth))
    ).mean().item()
    # top-5 is ill-defined for a 1-token predictor; report top1 acc as top5
    # (predictor can't do better).
    out["copy_left"] = {
        "loss": loss_copy,
        "ppl": float(torch.exp(torch.tensor(loss_copy)).item()),
        "top1": top1,
        "top5": top1,
        "n_masked": n_pos,
    }

    return out
