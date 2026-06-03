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

## Data: Garden-Path (GP) and Relative Clause (RC)

The GP and RC experiments use human self-paced reading times from the
**SAP Benchmark** (Huang et al., 2024). Download these files from the SAP
Benchmark
[Google Drive folder](https://drive.google.com/drive/folders/1g-oyH-XuB2oolo1d8KZfuFtiimuNyhjc)
into `data/` (large files, gitignored — download via the browser):

| File | Used by | Purpose |
|------|---------|---------|
| `ClassicGardenPathSet.csv` | GP | Garden-path (MVRR, NP/S, NP/Z) reading times |
| `Fillers.csv` | GP + RC eval | Fillers for fitting the reading-time regression |
| `RelativeClauseSet.csv` | RC | Relative-clause (RC_Subj, RC_Obj) reading times |

```bash
mkdir -p data   # then place the three CSVs above into data/
```

The preprocessing scripts create their own output directories. Preprocess from the
repo root (GP outputs under `src/garden_path_cross_validation/`,
RC under `src/relative_clause_cross_validation/`):

```bash
# Garden-path
uv run python src/garden_path_cross_validation/data_preparation/cli.py process-human-data --min-rt 100 --max-rt 3000
uv run python src/garden_path_cross_validation/data_preparation/cli.py create-folds --num-folds 24
uv run python src/garden_path_cross_validation/data_preparation/cli.py filter-folds

# Relative clause
uv run python src/relative_clause_cross_validation/data_preparation/cli.py process-human-data --min-rt 100 --max-rt 3000
uv run python src/relative_clause_cross_validation/data_preparation/cli.py create-folds --num-folds 24
uv run python src/relative_clause_cross_validation/data_preparation/cli.py filter-folds
```

Each produces subject averages and `fold_0 … fold_23` (23-train / 1-test) under
`folds_filtered/`. Both **evaluations** also need the processed fillers (the shared
regression baseline), prepared once:

```bash
uv run python src/garden_path_cross_validation/data_preparation/cli.py process-fillers --min-rt 100 --max-rt 3000
# -> src/garden_path_cross_validation/folds/fillers_processed.csv (used by GP and RC eval)
```
