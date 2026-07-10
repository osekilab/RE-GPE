# Reverse-Engineering Garden-Path Effects

Official implementation of the ACL 2026 paper *An Existence Proof for Neural
Language Models That Can Explain Garden-Path Effects via Surprisal*
(https://aclanthology.org/2026.acl-long.1694/) — code for reverse-engineering
human garden-path effects via language-model surprisal, built on
[`samuki/reverse-engineering-the-reader`](https://github.com/samuki/reverse-engineering-the-reader).

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

## Data: Naturalistic delta-LLH evaluation corpora

Three self-paced reading corpora are used **for evaluation only** (no model is
trained on them) to measure the delta log-likelihood of human reading times. Each
is processed into a `*_processed.csv` under `data/`.

### Natural Stories

From the public [Natural Stories corpus](https://github.com/languageMIT/naturalstories)
(a submodule):

```bash
git submodule update --init external/naturalstories
uv run python src/natural_stories_preparation/cli.py \
  --naturalstories-path external/naturalstories --output-dir data/
# -> data/natural_stories_processed.csv
```

### UCL

Self-paced reading subset of the UCL corpus (Frank et al., 2013). Download the
paper's supplementary material and unzip into `data/ucl/`:

```bash
mkdir -p data/ucl
curl -L -o /tmp/ucl_data.zip \
  "https://static-content.springer.com/esm/art%3A10.3758%2Fs13428-012-0313-y/MediaObjects/13428_2012_313_MOESM1_ESM.zip"
unzip -j /tmp/ucl_data.zip -d data/ucl
uv run python src/ucl_preparation/cli.py \
  --ucl-path data/ucl --output-dir data/ \
  --use-selfpaced --output-filename ucl_selfpaced_processed.csv
# -> data/ucl_selfpaced_processed.csv
```

### Smith 2013

Brown-corpus self-paced reading (Smith & Levy, 2013), read from the `data.pkl`
bundle redistributed in de Varda's
[`local_attention_reading_times`](https://github.com/Andrea-de-Varda/local_attention_reading_times).

```bash
curl -L -o data/data.pkl \
  https://raw.githubusercontent.com/Andrea-de-Varda/local_attention_reading_times/main/data.pkl
# expected md5: fa09cf375176fdde680d1ce47d7a2806
uv run python src/smith2013_preparation/cli.py \
  --smith2013-path data/data.pkl --output-dir data/
# -> data/smith2013_processed.csv
```

## Train

One model is fine-tuned per fold (leave-one-out: 23 training items, 1 held out).
Run from `src/reverse_engineering/`; the orchestrators generate a per-fold config
from a template and run `run.py` for each fold. The templates encode the paper's
hyperparameters and default to **GPT-2 small** (seed 42); for the other sizes
change `model:` to `"gpt2-medium"` / `"gpt2-large"`. On Apple Silicon, prefix
commands with `PYTORCH_ENABLE_MPS_FALLBACK=1`.

```bash
cd src/reverse_engineering

# Garden-path, all phenomena together (main)
uv run python train_all_valid_folds.py --output-dir output/gp_all

# Garden-path, one phenomenon at a time (analysis)
uv run python train_all_valid_folds.py \
  --base-config configs/garden_path_template_mvrr.yaml --output-dir output/gp_mvrr
# likewise _nps.yaml -> output/gp_nps, _npz.yaml -> output/gp_npz

# Relative clause
uv run python train_all_valid_folds_rc.py --output-dir output/rc
```

Each fold's model is saved under `<output-dir>/<construction>_fold_N/final_checkpoint`.
Useful options: `--folds 0 1 2` or `--start-fold` / `--end-fold` for a subset,
`--device cuda:0`, `--dry-run` to preview the commands.

## Evaluate

Each trained fold is evaluated for two things and compared against the untrained
baseline (the same GPT-2 from Hugging Face): the **garden-path / relative-clause
effect** (Amb − Unamb reading time at ROI 0/1/2, via a regression fit on the
fillers) and the **delta log-likelihood** on the naturalistic corpora. Run from
`src/reverse_engineering/`; `--config-template` must match the training template.
WT decoding is on by default (`--no-wt-decoding` to disable).

```bash
cd src/reverse_engineering

# Garden-path: trained folds + untrained baseline
uv run python evaluate_all_folds.py \
  --models-dir output/gp_all --config-template configs/garden_path_template.yaml \
  --summary-output all_folds_evaluation_summary.json \
  --eval-natural-stories --eval-ucl --eval-smith2013 --device cuda:0
uv run python evaluate_baseline.py all \
  --config-template configs/garden_path_template.yaml \
  --output-dir baseline_evaluation \
  --eval-natural-stories --eval-ucl --eval-smith2013 --device cuda:0

# Relative clause
uv run python evaluate_all_folds_rc.py \
  --models-dir output/rc --config-template configs/relative_clause_template.yaml \
  --folds-dir ../../src/relative_clause_cross_validation/folds_filtered \
  --summary-output all_folds_evaluation_summary_rc.json \
  --eval-natural-stories --eval-ucl --eval-smith2013 --device cuda:0
uv run python evaluate_baseline_rc.py all \
  --config-template configs/relative_clause_template.yaml \
  --folds-dir ../../src/relative_clause_cross_validation/folds_filtered \
  --output-dir baseline_evaluation_rc \
  --eval-natural-stories --eval-ucl --eval-smith2013 --device cuda:0
```

For the single-phenomenon runs, point `--models-dir` / `--config-template` at the
matching run (e.g. `output/gp_mvrr` with `configs/garden_path_template_mvrr.yaml`).
Each summary holds the per-construction Amb − Unamb effects (predicted and actual,
ROI 0/1/2) and the delta-LLH per corpus.

## Visualize

`visualize_results.py` produces the six figures reported in the paper:
`garden_path_effects_combined`, `delta_llh_combined`, `transfer_matrix_small_roi1`,
`transfer_matrix_all_sizes_roi1_md_lg_clean`, `rc_effects_combined`, and
`rc_delta_llh_combined`. The figures lay out GPT-2 small / medium / large side by
side, so all three sizes must be trained and evaluated first.

```bash
cd src/reverse_engineering
uv run python visualize_results.py \
  --summary-small  all_folds_evaluation_summary.json \
  --summary-medium all_folds_evaluation_summary_md.json \
  --summary-large  all_folds_evaluation_summary_lg.json \
  --baseline-small  baseline_evaluation/baseline_summary.json \
  --baseline-medium baseline_evaluation_md/baseline_summary.json \
  --baseline-large  baseline_evaluation_lg/baseline_summary.json \
  --output-dir figures
```

The RC figures read the RC summaries (`all_folds_evaluation_summary_rc{,_md,_lg}.json`
and `baseline_evaluation_rc*/baseline_summary.json`). The transfer-matrix figures read
the per-construction summaries from `--base-dir` (default `.`), expecting
`all_folds_evaluation_summary_{mvrr,nps,npz}{,_md,_lg}.json` — so name the
single-phenomenon eval outputs accordingly (e.g. `--summary-output
all_folds_evaluation_summary_mvrr.json` for the `output/gp_mvrr` run). The RC-Large
folds that diverged (20, 22) are dropped from the aggregates automatically. Figures
are written to `figures/`.

## Appendix

### Perplexity

Subword perplexity on the naturalistic corpora for the trained folds and the
untrained baseline, in one run:

```bash
cd src/reverse_engineering
uv run python evaluate_perplexity.py --all-folds --folds-dir output/gp_all \
  --baseline-model gpt2 --output-path perplexity_evaluation/results.json
# RC: --folds-dir output/rc --fold-pattern "relative_clause_fold_*" \
#     --output-path perplexity_evaluation/results_rc.json
```

The baseline is evaluated in the same run (`--skip-baseline` to skip). For GPT-2
medium/large, set `--baseline-model` and `--folds-dir` accordingly.

### BLiMP

BLiMP accuracy via the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness),
aggregated across folds. Requires the optional `blimp` extra:

```bash
uv sync --extra blimp
# from the repo root (runs lm_eval on the baseline and each fold, then aggregates):
bash scripts/evaluate_blimp_gp.sh   # -> src/reverse_engineering/blimp_evaluation/summary.json
bash scripts/evaluate_blimp_rc.sh   # RC (reuses the baseline from the GP script)
```

Override `MODELS_DIR` / `BASELINE_MODEL` / `OUT` (and `BLIMP_DEVICE`,
`BLIMP_BATCH_SIZE`) for GPT-2 medium/large.

### Perplexity + BLiMP table

Once the perplexity (`perplexity_evaluation/results*.json`) and BLiMP
(`blimp_evaluation*/summary.json`) results exist for all three sizes, render the
appendix LaTeX table (`tab:ppl_blimp_appendix`):

```bash
cd src/reverse_engineering
uv run python generate_appendix_table.py   # -> figures/ppl_blimp_appendix.tex
```

It covers Small / Medium / Large (see the `SIZES` list for the expected result
filenames); the RC-Large folds that failed to converge (20, 22) are excluded
automatically.

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{yoshida-etal-2026-existence,
    title = "An Existence Proof for Neural Language Models That Can Explain Garden-Path Effects via Surprisal",
    author = "Yoshida, Ryo  and
      Isono, Shinnosuke  and
      Someya, Taiga  and
      Oseki, Yohei  and
      Kuribayashi, Tatsuki",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.1694/",
    doi = "10.18653/v1/2026.acl-long.1694",
    pages = "36556--36569",
    ISBN = "979-8-89176-390-6",
    abstract = "Surprisal theory hypothesizes that the difficulty of human sentence processing increases linearly with surprisal, the negative log-probability of a word given its context. Computational psycholinguistics has tested this hypothesis using language models (LMs) as proxies for human prediction. While surprisal derived from recent neural LMs generally captures human processing difficulty on naturalistic corpora that predominantly consist of simple sentences, it severely underestimates processing difficulty on sentences that require syntactic disambiguation (garden-path effects). This leads to the claim that the processing difficulty of such sentences cannot be reduced to surprisal, although it remains possible that neural LMs simply differ from humans in next-word prediction. In this paper, we investigate whether it is truly impossible to construct a neural LM that can explain garden-path effects via surprisal. Specifically, instead of evaluating off-the-shelf neural LMs, we fine-tune these LMs on garden-path sentences so as to better align surprisal-based reading-time estimates with actual human reading times. Our results show that fine-tuned LMs do not overfit and successfully capture human reading slowdowns on held-out garden-path items; they even improve predictive power for human reading times on naturalistic corpora and preserve their general LM capabilities. These results provide an existence proof for a neural LM that can explain both garden-path effects and naturalistic reading times via surprisal, but also raise a theoretical question: what kind of evidence can truly falsify surprisal theory?"
}
```
