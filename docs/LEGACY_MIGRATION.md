# Legacy preprocessing migration

This document records how the ChromBERT preprocessing code was pulled into
FlashChromBert, so the next person doing a similar port can follow the same
rules without guessing.

**TL;DR** — the old `chrombert_utils/css_utility.py` lives under
`src/flashchrombert/legacy/` **byte-identical** to the source. All new logic
(tokenizer, masking, data pipeline, CLI) is a sibling adapter that imports
from it. The legacy file is never edited.

---

## 1. Scope and rule

**Scope.** The research asset worth preserving is the preprocessing /
chromatin-state pipeline in `ChromBERT/processing/chrombert_utils/css_utility.py`
(3306 lines of BED parsing, unit-CSS conversion, k-mer generation, promoter
extraction, motif utilities, plotting). Everything else in the old ChromBERT
repo (bert, trainer, Apex FP16, HuggingFace 2020-era fork) is commodity and
was rewritten.

**Rule.** The legacy file is copied in **verbatim** (confirmed with
`diff -q`). It is never edited. Adaptations — renames, type hints, lazy
imports, API cleanup — go into new sibling modules. This keeps upstream
parity obvious, decouples the new project from the old repo's lifecycle,
and sidesteps the old dependency stack (matplotlib / tslearn / umap-learn /
pybedtools / biopython / logomaker / wordcloud / stylecloud / …) unless a
caller actually exercises those functions.

Provenance: copied from ChromBERT commit `8506b27b` (2025-12-16).

---

## 2. Repository layout

```
FlashChromBert/
├── src/flashchrombert/
│   ├── legacy/
│   │   ├── __init__.py          # `from .css_utility import *`
│   │   └── css_utility.py       # verbatim copy, DO NOT EDIT
│   └── data/
│       ├── tokenizer.py         # adapters: CStateTokenizer, KmerCStateTokenizer
│       ├── dataset.py           # adapters: KmerMaskListMaskingStrategy (MASK_LIST)
│       └── datamodule.py        # DataModules that accept any MaskingStrategy
├── configs/
│   ├── kmer_legacy.yaml         # exercises the legacy vocab + MASK_LIST
│   └── tiny_css.yaml            # fixed-length synthetic chromatin states
├── tests/
│   └── test_kmer_legacy.py      # vocab parity + seq2kmer parity (latter opt-in)
└── pyproject.toml               # [legacy] extra holds the heavy deps
```

---

## 3. Step-by-step record

### 3.1 Copy the legacy file

```bash
cp ChromBERT/processing/chrombert_utils/css_utility.py \
   FlashChromBert/src/flashchrombert/legacy/css_utility.py
diff -q <source> <dest>          # must report no differences
```

Then a one-line `__init__.py` that re-exports everything:

```python
from .css_utility import *  # noqa: F401,F403
```

### 3.2 Quarantine the heavy deps

The verbatim file imports matplotlib, seaborn, scipy, tslearn, sklearn,
networkx, umap, biopython, logomaker, wordcloud, pandas, statsmodels,
seqeval, pyahocorasick, sentencepiece, pybedtools, stylecloud at module
top. None of those are in the default install. Add a `[legacy]` extra:

```toml
[project.optional-dependencies]
legacy = [
    "pandas>=2.2", "matplotlib>=3.8", "seaborn>=0.13", "scipy>=1.12",
    "scikit-learn>=1.4", "networkx>=3.2", "tslearn>=0.6", "umap-learn>=0.5",
    "biopython", "logomaker>=0.8", "wordcloud>=1.9", "stylecloud",
    "statsmodels>=0.14", "seqeval>=1.2", "pyahocorasick",
    "sentencepiece>=0.1.99", "pybedtools>=0.9",
]
```

Install only when you actually run legacy functions:

```bash
pip install -e .[legacy]
```

Base tests (`pytest tests/`) must stay green without this extra. The one
parity test that imports `chrombert_utils` uses `pytest.importorskip` so it
cleanly skips when the stack is missing.

### 3.3 Identify the adapter surface

Grepped `chrombert_utils/css_utility.py` for `^def ` and picked the subset
that the BERT pretraining path actually needs:

| Legacy function            | Purpose                                                  |
|----------------------------|----------------------------------------------------------|
| `state_dict`, `state_dict_18` | state number → letter (A..O / A..R)                   |
| `seq2kmer(seq, k, stride)` | chromatin state string → space-separated k-mer string   |
| `kmer2seq(kmers)`          | inverse                                                  |
| `bed2df_expanded`          | ROADMAP/IHEC BED → DataFrame                             |
| `df2longcss`, `df2unitcss` | DataFrame → per-chromosome unit-CSS                      |
| `save_css_by_cell_wo_continuous_{15,18}state` | per-cell pickled k-mer text     |
| `kmerCSS_to_pretrain_data_ihec` | concatenate cells into a pretrain `.txt`            |

The mask-aware MLM rule (`MASK_LIST`) does **not** live in
`chrombert_utils`; it was embedded in `ChromBERT/training/examples/run_pretrain.py`
(lines 100–105, 262–312). Because that's an orchestration script, not a
library, I ported it into FlashChromBert rather than copying the script.

### 3.4 Port the tokenizer

`KmerCStateTokenizer(k, num_states)` in
`src/flashchrombert/data/tokenizer.py`.

- Vocab = `itertools.product(letters, repeat=k)` where
  `letters = "ABCDEFGHIJKLMNO"` (15 states) or `"ABCDEFGHIJKLMNOPQR"`
  (18 states). Special tokens `[PAD][UNK][CLS][SEP][MASK]` are always
  prepended at ids 0..4 (contract from `Tokenizer` base class).
- `_split(text)` is `text.split()` — matches the whitespace-separated k-mer
  layout emitted by the legacy `seq2kmer`.
- `from_vocab_file(path, k, num_states)` classmethod loads
  `ChromBERT/training/src/transformers/dnabert-config/bert-config-{k}/vocab.txt`
  and guarantees id-identical round-trip.

**Legacy quirk caught during port.** The 18-state k=4 legacy vocab excludes
`RRRR` (see `bert-config-18-4/generate_4mer.py`), because state R (Quies in
18-state) dominates runs of non-informative genome. The 15-state k=4 vocab
does **not** exclude `OOOO` — the training text just strips it via
`save_css_by_cell_wo_continuous_15state`. To match both files without
special-casing, the constructor auto-excludes `RRRR` when `num_states=18`
and `k=4`, and accepts a general `exclude_tokens` kwarg for any future
variant. This made `KmerCStateTokenizer(k=4, num_states=18)` equal the
legacy `bert-config-18-4/vocab.txt` without extra args.

### 3.5 Port the masking rule

`KmerMaskListMaskingStrategy(k, mlm_probability)` in
`src/flashchrombert/data/dataset.py`, behaviour identical to
`mask_tokens` in `run_pretrain.py`:

1. Draw 15% centers with Bernoulli, zeroing probability on specials and pad.
2. Expand each center `c` with offsets from `MASK_LIST[k]`
   (e.g. `k=4 → [-1, 1, 2]`) clamped to `[1, end]`, where `end` is the last
   non-zero-probability index.
3. Set `labels[~masked] = -100`.
4. 80% of masked positions → `[MASK]`, 10% → random token, 10% unchanged.

This is the adapter **for** the legacy masking semantics; the legacy file
itself was never touched.

### 3.6 Let SDPA actually pick FlashAttention

Independent of the legacy code, there's a transport-layer detail: PyTorch
2.5 SDPA refuses FlashAttention when `attn_mask is not None`. For
equal-length batches (typical for chunked genome / k-mer sequences there
is no padding), `collate_mlm` now emits `attention_mask=None`, letting
SDPA dispatch to FA2 on Ada+. Verified via:

```python
from torch.nn.attention import SDPBackend, sdpa_kernel
with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    out = lit.model(**batch)                # must not fall back
```

### 3.7 Parity verification

Two parity tiers in `tests/test_kmer_legacy.py`:

1. **Id-table parity (always runs).** Enumerated vocab equals the legacy
   `bert-config-{3,4,5,6}/vocab.txt` and `bert-config-18-4/vocab.txt`
   byte-for-byte. Only needs the legacy files on disk, no heavy deps.
2. **End-to-end seq2kmer parity (opt-in).** Feeds a known state string
   through `flashchrombert.legacy.css_utility.seq2kmer` and through
   `KmerCStateTokenizer.encode`, asserts id-sequence equality and full
   roundtrip. Uses `pytest.importorskip("flashchrombert.legacy.css_utility")`
   so it skips cleanly without the `[legacy]` extra installed.

Plus MASK_LIST expansion sanity checks (centers expand within bounds,
specials and pad stay at -100).

### 3.8 Config + CLI wiring

`configs/kmer_legacy.yaml` exercises the full path: legacy-id vocab,
`kmer_mask_list` masking, `random_fixed` dataset for isolated benchmarking,
bf16-mixed precision. The same config template plugs in a real legacy
pretrain `.txt` by switching `data.kind: file` and pointing `train_file`
at the output of `kmerCSS_to_pretrain_data_ihec`.

CLI `build_tokenizer` now dispatches `kmer_cstate` (with optional
`vocab_file:` to pin to the legacy id table) and `build_masking` dispatches
`kmer_mask_list`.

---

## 4. How to add the next legacy module

The same pattern generalises. When pulling in another module from the old
repo (say `motif_utils.py`):

1. `cp <old>/motif_utils.py src/flashchrombert/legacy/motif_utils.py` —
   verbatim, verify with `diff -q`.
2. Re-export from `legacy/__init__.py`.
3. Add any new transitive deps to the `[legacy]` extra in `pyproject.toml`.
4. Write an adapter under `src/flashchrombert/<area>/` that imports the
   legacy functions lazily (inside a function body, not at module top)
   if the caller path can skip the heavy stack.
5. Golden-fixture test alongside existing ones, guarded with
   `pytest.importorskip`.

**Do not** refactor the legacy file, even for obvious issues (unused
imports, deprecated `SyntaxWarning: invalid escape sequence '\d'` at
line 2886, etc.). If a bug matters, fix it in an adapter — never in the
verbatim file. This rule is absolute: the legacy file must remain a
bit-perfect snapshot of upstream so future re-syncs are a diff, not a
merge.

---

## 5. Commands cheat sheet

```bash
# Verify legacy copy is identical to upstream
diff -q ChromBERT/processing/chrombert_utils/css_utility.py \
        FlashChromBert/src/flashchrombert/legacy/css_utility.py

# Run base tests (no heavy deps)
pytest tests/ -q                              # 19 passed, 1 skipped

# Run parity tests including seq2kmer round-trip
pip install -e .[legacy]
pytest tests/test_kmer_legacy.py -v

# Train a tiny run on the legacy k-mer vocab (FA2 + bf16)
fcbert-pretrain --config configs/kmer_legacy.yaml
```
