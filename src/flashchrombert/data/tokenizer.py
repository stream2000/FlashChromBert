from __future__ import annotations

import itertools
import string
from abc import ABC, abstractmethod
from pathlib import Path

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

# Legacy ChromBERT state → letter mapping (state_dict / state_dict_18 in css_utility.py).
STATE_LETTERS_15 = "ABCDEFGHIJKLMNO"       # 15 ROADMAP states
STATE_LETTERS_18 = "ABCDEFGHIJKLMNOPQR"    # 18 IHEC states


class Tokenizer(ABC):
    """Minimal tokenizer interface. Subclass for char / k-mer / BPE / etc.

    Contract: vocab ids 0..4 are reserved for special tokens (PAD/UNK/CLS/SEP/MASK).
    """

    def __init__(self, vocab: list[str]):
        # Prepend specials, dedup-preserving-order
        all_tokens = SPECIAL_TOKENS + [t for t in vocab if t not in SPECIAL_TOKENS]
        self.id_to_token = all_tokens
        self.token_to_id = {t: i for i, t in enumerate(all_tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["[PAD]"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["[UNK]"]

    @property
    def cls_token_id(self) -> int:
        return self.token_to_id["[CLS]"]

    @property
    def sep_token_id(self) -> int:
        return self.token_to_id["[SEP]"]

    @property
    def mask_token_id(self) -> int:
        return self.token_to_id["[MASK]"]

    def special_token_ids(self) -> set[int]:
        return {self.token_to_id[t] for t in SPECIAL_TOKENS}

    @abstractmethod
    def _split(self, text: str) -> list[str]: ...

    def encode(self, text: str, add_special: bool = True) -> list[int]:
        ids = [self.token_to_id.get(t, self.unk_token_id) for t in self._split(text)]
        if add_special:
            ids = [self.cls_token_id] + ids + [self.sep_token_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        tokens = [self.id_to_token[i] for i in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in SPECIAL_TOKENS]
        return self._join(tokens)

    def _join(self, tokens: list[str]) -> str:
        return "".join(tokens)


class CharTokenizer(Tokenizer):
    """Character-level tokenizer for text pretraining demos.

    Extend by swapping in KmerTokenizer later — BertModel/Lightning code unchanged.
    """

    def __init__(self, extra_chars: str = ""):
        base = string.ascii_letters + string.digits + string.punctuation + " \n\t"
        vocab = list(dict.fromkeys(base + extra_chars))
        super().__init__(vocab)

    def _split(self, text: str) -> list[str]:
        return list(text)


class CStateTokenizer(Tokenizer):
    """Whitespace-separated chromatin-state tokenizer (ROADMAP 15-state by default).

    A line looks like: "E1 E2 E15 E3 ..." — one token per position. The k-mer
    variant used downstream will subclass this and override `_split`.
    """

    def __init__(self, num_states: int = 15):
        vocab = [f"E{i}" for i in range(1, num_states + 1)]
        super().__init__(vocab)

    def _split(self, text: str) -> list[str]:
        return text.split()

    def _join(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class KmerCStateTokenizer(Tokenizer):
    """Legacy ChromBERT k-mer tokenizer.

    Vocab is the full cartesian product of the state alphabet (A..O for 15-state,
    A..R for 18-state) at length `k`, in the same order as ChromBERT's
    `training/src/transformers/dnabert-config/bert-config-{k}/vocab.txt`.

    Inputs are whitespace-separated k-mer strings produced by
    `chrombert_utils.seq2kmer` (preserved verbatim in `flashchrombert.legacy`).
    """

    def __init__(
        self,
        k: int,
        num_states: int = 15,
        exclude_tokens: list[str] | None = None,
        *,
        _vocab_override: list[str] | None = None,
    ):
        if num_states == 15:
            letters = STATE_LETTERS_15
        elif num_states == 18:
            letters = STATE_LETTERS_18
        else:
            raise ValueError(f"Only 15 or 18 states are supported, got {num_states}")
        self.k = k
        self.num_states = num_states

        # Legacy quirk: bert-config-18-4/generate_4mer.py excludes RRRR from the
        # 18-state k=4 vocab. Default here matches that, so `from_vocab_file`
        # round-trips identically.
        if exclude_tokens is None and num_states == 18 and k == 4:
            exclude_tokens = ["RRRR"]
        excluded = set(exclude_tokens or [])

        if _vocab_override is None:
            vocab = [
                "".join(p)
                for p in itertools.product(letters, repeat=k)
                if "".join(p) not in excluded
            ]
        else:
            vocab = _vocab_override
        super().__init__(vocab)

    def _split(self, text: str) -> list[str]:
        return text.split()

    def _join(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    @classmethod
    def from_vocab_file(
        cls, path: str | Path, k: int, num_states: int = 15
    ) -> KmerCStateTokenizer:
        """Load the legacy vocab.txt and return an id-identical tokenizer."""
        tokens: list[str] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                t = line.rstrip("\n")
                if t:
                    tokens.append(t)
        if tokens[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError(
                f"{path}: first {len(SPECIAL_TOKENS)} lines must be {SPECIAL_TOKENS}"
            )
        kmers = tokens[len(SPECIAL_TOKENS) :]
        return cls(k=k, num_states=num_states, _vocab_override=kmers)
