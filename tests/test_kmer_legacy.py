"""Parity tests for the k-mer adapter against the legacy ChromBERT pipeline.

Two kinds of checks:
  1. Pure id-table parity — the KmerCStateTokenizer enumeration must equal the
     legacy `bert-config-{k}/vocab.txt` files byte-for-byte. No heavy deps needed.
  2. seq2kmer parity — feed a known chromatin-state string through the verbatim
     `chrombert_utils.seq2kmer` (copied into `flashchrombert.legacy`) and through
     our tokenizer; token ids must match. This requires pandas/matplotlib/etc.
     (pip install -e .[legacy]) and is skipped otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from flashchrombert.data import KmerCStateTokenizer, KmerMaskListMaskingStrategy

LEGACY_VOCAB_DIR = Path(
    "/home/fqijun/python/ChromBERT/training/src/transformers/dnabert-config"
)


@pytest.mark.parametrize(
    "k,num_states,subdir",
    [
        (3, 15, "bert-config-3"),
        (4, 15, "bert-config-4"),
        (5, 15, "bert-config-5"),
        (6, 15, "bert-config-6"),
        (4, 18, "bert-config-18-4"),
    ],
)
def test_kmer_vocab_matches_legacy(k, num_states, subdir):
    vocab_path = LEGACY_VOCAB_DIR / subdir / "vocab.txt"
    if not vocab_path.exists():
        pytest.skip(f"legacy vocab missing at {vocab_path}")
    ours = KmerCStateTokenizer(k=k, num_states=num_states)
    legacy = KmerCStateTokenizer.from_vocab_file(vocab_path, k=k, num_states=num_states)
    assert ours.id_to_token == legacy.id_to_token


def test_legacy_seq2kmer_parity():
    """End-to-end: chromatin state string -> legacy seq2kmer -> our tokenizer ids.

    Requires the `legacy` extra (pandas, matplotlib, ...). Skipped otherwise.
    """
    try:
        from flashchrombert.legacy.css_utility import seq2kmer
    except ImportError as e:
        pytest.skip(f"legacy deps not installed ({e}); run `pip install -e .[legacy]`")

    seq = "ABCDEFGHIJKLMNOACGIKMOABCDEFGHIJKLMNO"
    k = 4
    kmer_text = seq2kmer(seq, k)
    tok = KmerCStateTokenizer(k=k, num_states=15)
    ids = tok.encode(kmer_text, add_special=True)
    # Spot-check: first and last content ids decode to the first/last k-mers
    assert ids[0] == tok.cls_token_id
    assert ids[-1] == tok.sep_token_id
    assert tok.id_to_token[ids[1]] == seq[:k]
    assert tok.id_to_token[ids[-2]] == seq[-k:]
    # Full roundtrip
    assert tok.decode(ids, skip_special=True) == kmer_text


def test_mask_list_scheme_expands_centers():
    torch.manual_seed(0)
    tok = KmerCStateTokenizer(k=4, num_states=15)
    strat = KmerMaskListMaskingStrategy(k=4, mlm_probability=1.0)  # force every token
    seq = torch.tensor(
        [[tok.cls_token_id] + list(range(100, 120)) + [tok.sep_token_id]]
    )
    inputs, labels = strat.mask(seq, tok)
    # Specials (positions 0 and 21) must be ignored.
    assert labels[0, 0] == -100
    assert labels[0, 21] == -100
    # All interior positions must be labelled since prob=1.0 selects every content token
    # AND MASK_LIST expansion can only widen the set.
    interior = labels[0, 1:21]
    assert (interior != -100).all()


def test_mask_list_scheme_respects_pad():
    torch.manual_seed(0)
    tok = KmerCStateTokenizer(k=4, num_states=15)
    strat = KmerMaskListMaskingStrategy(k=4, mlm_probability=1.0)
    # Last 5 positions are pads.
    seq = torch.tensor(
        [[tok.cls_token_id] + list(range(100, 115)) + [tok.sep_token_id]
         + [tok.pad_token_id] * 5]
    )
    _, labels = strat.mask(seq, tok)
    # Pad positions must keep label -100.
    assert (labels[0, -5:] == -100).all()
