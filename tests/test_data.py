import torch
from unittest.mock import MagicMock

from flashchrombert.data import (
    CharTokenizer,
    KmerCStateTokenizer,
    MLMDataset,
    StandardMaskingStrategy,
)
from flashchrombert.data.dataset import StreamingMLMDataset, collate_mlm


def test_tokenizer_roundtrip():
    tok = CharTokenizer()
    text = "hello world"
    ids = tok.encode(text, add_special=True)
    assert ids[0] == tok.cls_token_id
    assert ids[-1] == tok.sep_token_id
    assert tok.decode(ids, skip_special=True) == text


def test_tokenizer_specials_reserved():
    tok = CharTokenizer()
    # First 5 ids must be the special tokens in fixed order
    assert tok.id_to_token[:5] == ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def test_dataset_reads_lines(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("hello\nworld\n\n  spaced  \n")
    tok = CharTokenizer()
    ds = MLMDataset(f, tok, max_length=32)
    assert len(ds) == 3  # empty line dropped
    assert isinstance(ds[0], torch.Tensor)


def test_masking_preserves_specials():
    torch.manual_seed(0)
    tok = CharTokenizer()
    strategy = StandardMaskingStrategy(mlm_probability=0.5)
    seq = torch.tensor(
        [[tok.cls_token_id, 10, 11, 12, tok.sep_token_id, tok.pad_token_id]]
    )
    _, labels = strategy.mask(seq, tok)
    # Special positions must have label -100 (not chosen for masking)
    assert labels[0, 0] == -100
    assert labels[0, 4] == -100
    assert labels[0, 5] == -100


def test_collate_pads_to_max_len_in_batch():
    tok = CharTokenizer()
    strategy = StandardMaskingStrategy()
    batch = [
        torch.tensor([tok.cls_token_id, 10, 11, tok.sep_token_id]),
        torch.tensor([tok.cls_token_id, 20, 21, 22, 23, tok.sep_token_id]),
    ]
    out = collate_mlm(batch, tokenizer=tok, masking=strategy, max_length=128)
    assert out["input_ids"].shape == (2, 6)
    assert out["attention_mask"].shape == (2, 6)
    assert out["attention_mask"][0].tolist() == [1, 1, 1, 1, 0, 0]
    assert out["attention_mask"][1].tolist() == [1, 1, 1, 1, 1, 1]


# ---------------------------------------------------------------------------
# StreamingMLMDataset tests
# ---------------------------------------------------------------------------

def test_streaming_dataset_yields_tensors(tmp_path):
    f = tmp_path / "data.txt"
    # 4-mer kmer_cstate format: space-separated 4-char tokens
    tok = KmerCStateTokenizer(k=4, num_states=15)
    line = " ".join(["AAAA"] * 10)
    f.write_text(f"{line}\n{line}\n\n")  # two content lines, one empty

    ds = StreamingMLMDataset(f, tok, max_length=12)
    items = list(ds)
    assert len(items) == 2  # one chunk per line (10 tokens < chunk_size=10)
    for item in items:
        assert isinstance(item, torch.Tensor)
        assert item[0] == tok.cls_token_id
        assert item[-1] == tok.sep_token_id
        assert item.max() < tok.vocab_size


def test_streaming_dataset_chunks_long_lines(tmp_path):
    f = tmp_path / "data.txt"
    tok = KmerCStateTokenizer(k=4, num_states=15)
    # max_length=6 → chunk_size=4; 12 tokens → 3 chunks
    line = " ".join(["AAAA"] * 12)
    f.write_text(line + "\n")

    ds = StreamingMLMDataset(f, tok, max_length=6)
    items = list(ds)
    assert len(items) == 3  # ceil(12 / 4) = 3
    for item in items:
        assert item[0] == tok.cls_token_id
        assert item[-1] == tok.sep_token_id


def test_streaming_dataset_worker_sharding(tmp_path):
    """Each worker must see non-overlapping lines; union must be full dataset."""
    f = tmp_path / "data.txt"
    tok = KmerCStateTokenizer(k=4, num_states=15)
    lines = [" ".join(["AAAA"] * 5)] * 6  # 6 lines
    f.write_text("\n".join(lines) + "\n")

    ds = StreamingMLMDataset(f, tok, max_length=512)

    def _iter_as_worker(worker_id, num_workers):
        info = MagicMock()
        info.id = worker_id
        info.num_workers = num_workers
        import torch.utils.data
        original = torch.utils.data.get_worker_info
        torch.utils.data.get_worker_info = lambda: info
        try:
            return list(ds)
        finally:
            torch.utils.data.get_worker_info = original

    items_w0 = _iter_as_worker(0, 2)
    items_w1 = _iter_as_worker(1, 2)
    # No overlap in line count: each worker gets exactly 3 of 6 lines
    assert len(items_w0) == 3
    assert len(items_w1) == 3


# ---------------------------------------------------------------------------
# KmerCStateTokenizer 18-state correctness
# ---------------------------------------------------------------------------

def test_kmer_18state_vocab_size():
    tok = KmerCStateTokenizer(k=4, num_states=18)
    # 18^4 = 104976, minus RRRR = 104975, plus 5 specials = 104980
    assert tok.vocab_size == 104980


def test_kmer_18state_excludes_rrrr():
    tok = KmerCStateTokenizer(k=4, num_states=18)
    assert "RRRR" not in tok.token_to_id


def test_kmer_18state_encodes_valid_token():
    tok = KmerCStateTokenizer(k=4, num_states=18)
    # ABCD is a valid 18-state k=4 token
    ids = tok.encode("ABCD", add_special=False)
    assert len(ids) == 1
    assert ids[0] >= 5  # not a special token id
    assert ids[0] < tok.vocab_size
