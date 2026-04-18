# FlashChromBert — Project Notes

## Environment

Conda env: `flashchrombert` (Python 3.12).

Dependencies: `torch` (CUDA 12.4 build), `lightning`, `pyyaml`, `numpy`, `tqdm`, `pytest`.

### How it was built

```bash
source /work/miniconda3/etc/profile.d/conda.sh
conda create -n flashchrombert python=3.12 -y
conda activate flashchrombert

# torch must match the system CUDA driver (12.4 on this cluster)
pip install --index-url https://download.pytorch.org/whl/cu124 "torch==2.5.*"
pip install lightning pyyaml numpy tqdm pytest

# Project in editable mode
pip install -e .
```

### Activation

Use `./activate.sh` at repo root — it handles the library-path workaround below.

---

## Known issue: `libnvJitLink` symbol error

### Symptom

```
ImportError: .../torch/lib/../../nvidia/cusparse/lib/libcusparse.so.12:
  undefined symbol: __nvJitLinkComplete_12_4, version libnvJitLink.so.12
```

### Cause

`pip install torch` bundles NVIDIA libraries under `site-packages/nvidia/*/lib/`. `libcusparse.so.12` shipped with torch requires a specific `libnvJitLink.so.12`, but the dynamic loader may pick up an older one from another path first (system CUDA, conda base, or another env).

**All the required files are present in the env** — only the library search order is wrong.

### Fix

Prepend the env's bundled NVIDIA library paths to `LD_LIBRARY_PATH` before importing torch. Handled by `activate.sh`:

```bash
NVIDIA_LIB_ROOT="$CONDA_PREFIX/lib/python3.12/site-packages/nvidia"
for lib_dir in "$NVIDIA_LIB_ROOT"/*/lib; do
    export LD_LIBRARY_PATH="$lib_dir:$LD_LIBRARY_PATH"
done
```

If you activate the env manually (without `activate.sh`), either source that script after `conda activate flashchrombert`, or set the path yourself before running Python.

---

## Running tests

```bash
./activate.sh
pytest tests/ -v
```

Smoke + overfit tests exercise CUDA paths if available; shape and masking tests run on CPU.

## Running a training demo

```bash
./activate.sh
fcbert-pretrain --config configs/tiny_text.yaml
```
