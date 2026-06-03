#!/usr/bin/env bash
#
# BLiMP evaluation for the garden-path models (GPT-2 small) and the untrained
# baseline, then aggregation into a single summary.json.
#
# Requires the optional 'blimp' dependency (lm-eval):
#     uv sync --extra blimp
#
# Run from the repository root. Override the defaults via environment variables:
#     MODELS_DIR  (default: output/gp_all)   trained-fold directory
#     BASELINE_MODEL (default: gpt2)         HF id of the untrained baseline
#     OUT         (default: blimp_evaluation)
#     BLIMP_BATCH_SIZE (default: 256), BLIMP_DEVICE (default: cuda:0)
#
set -e
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT/src/reverse_engineering"

BATCH_SIZE="${BLIMP_BATCH_SIZE:-256}"
DEVICE="${BLIMP_DEVICE:-cuda:0}"
MODELS_DIR="${MODELS_DIR:-output/gp_all}"
BASELINE_MODEL="${BASELINE_MODEL:-gpt2}"
OUT="${OUT:-blimp_evaluation}"

# Untrained baseline (shared with the relative-clause models).
if [ ! -d "$OUT/baseline" ]; then
  echo "=== BLiMP baseline: $BASELINE_MODEL ==="
  mkdir -p "$OUT/baseline"
  uv run lm_eval --model hf --model_args "pretrained=${BASELINE_MODEL}" \
    --tasks blimp --batch_size "$BATCH_SIZE" --device "$DEVICE" \
    --output_path "$OUT/baseline/"
fi

# Trained folds.
for fold in $(seq 0 22); do
  CKPT="${MODELS_DIR}/garden_path_fold_${fold}/final_checkpoint"
  if [ ! -d "$CKPT" ]; then
    echo "skip fold ${fold} (no checkpoint at ${CKPT})"
    continue
  fi
  echo "=== BLiMP fold ${fold} ==="
  mkdir -p "$OUT/fold_${fold}"
  uv run lm_eval --model hf --model_args "pretrained=${CKPT}" \
    --tasks blimp --batch_size "$BATCH_SIZE" --device "$DEVICE" \
    --output_path "$OUT/fold_${fold}/"
done

# Aggregate baseline + folds into one summary.
uv run python aggregate_blimp.py --blimp-dir "$OUT" --output-path "$OUT/summary.json"
echo "Done. Summary: src/reverse_engineering/${OUT}/summary.json"
