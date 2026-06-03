# Reverse-Engineering the Reader: Garden-Path Fine-tuning

<!-- TODO: intro/overview (refine later). -->

## Environment Setup

Requires **Python ≥ 3.10**, [**uv**](https://docs.astral.sh/uv/), and **git**.

```bash
git clone https://github.com/osekilab/RE-GPE.git
cd RE-GPE
uv venv      # create .venv
uv sync      # install locked dependencies (pyproject.toml / uv.lock)
```

Run commands through `uv` (e.g. `uv run python <script>.py`). On Apple Silicon,
prefix training/evaluation with `PYTORCH_ENABLE_MPS_FALLBACK=1` so MPS-unsupported
ops fall back to CPU.

Some data-preparation steps need git submodules; each is initialized in its own
section below.
