#!/usr/bin/env bash
#
# BLiMP evaluation for the relative-clause models (GPT-2 small), then aggregation.
# The untrained baseline is the same gpt2 as for the garden-path models, so it is
# reused from blimp_evaluation/baseline/ when available (otherwise it is run here).
#
# Requires the optional 'blimp' dependency (lm-eval):
#     uv sync --extra blimp
#
# Run from the repository root. Overridable env vars:
#     MODELS_DIR (default: output/rc), BASELINE_MODEL (default: gpt2),
#     OUT (default: blimp_evaluation_rc), BLIMP_BATCH_SIZE (256), BLIMP_DEVICE (cuda:0)
#
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT/src/reverse_engineering"

BATCH_SIZE="${BLIMP_BATCH_SIZE:-256}"
DEVICE="${BLIMP_DEVICE:-cuda:0}"
MODELS_DIR="${MODELS_DIR:-output/rc}"
BASELINE_MODEL="${BASELINE_MODEL:-gpt2}"
OUT="${OUT:-blimp_evaluation_rc}"

# Baseline: reuse the garden-path baseline if present, else compute it here.
if [ ! -d "$OUT/baseline" ]; then
  if [ -d "blimp_evaluation/baseline" ]; then
    echo "=== Reusing BLiMP baseline from blimp_evaluation/baseline ==="
    mkdir -p "$OUT"
    cp -r "blimp_evaluation/baseline" "$OUT/baseline"
  else
    echo "=== BLiMP baseline: $BASELINE_MODEL ==="
    mkdir -p "$OUT/baseline"
    uv run lm_eval --model hf --model_args "pretrained=${BASELINE_MODEL}" \
      --tasks blimp --batch_size "$BATCH_SIZE" --device "$DEVICE" \
      --output_path "$OUT/baseline/"
  fi
fi

# Trained folds.
for fold in $(seq 0 22); do
  CKPT="${MODELS_DIR}/relative_clause_fold_${fold}/final_checkpoint"
  if [ ! -d "$CKPT" ]; then
    echo "skip fold ${fold} (no checkpoint at ${CKPT})"
    continue
  fi
  echo "=== BLiMP RC fold ${fold} ==="
  mkdir -p "$OUT/fold_${fold}"
  uv run lm_eval --model hf --model_args "pretrained=${CKPT}" \
    --tasks blimp --batch_size "$BATCH_SIZE" --device "$DEVICE" \
    --output_path "$OUT/fold_${fold}/"
done

uv run python aggregate_blimp.py --blimp-dir "$OUT" --output-path "$OUT/summary.json"
echo "Done. Summary: src/reverse_engineering/${OUT}/summary.json"
