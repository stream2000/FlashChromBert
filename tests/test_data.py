import torch

from flashchrombert.data import CharTokenizer, MLMDataset, StandardMaskingStrategy
from flashchrombert.data.dataset import collate_mlm


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
