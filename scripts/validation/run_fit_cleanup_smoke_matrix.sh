#!/bin/bash
set -euo pipefail

if [[ ! -f fit_gmm.py || ! -f fit_tuned_lens.py || ! -d scripts ]]; then
  echo "Run this script from the repository root." >&2
  exit 2
fi

PHASE="${PHASE:-both}" # baseline | candidate | compare | both
SMOKE_ROOT="${SMOKE_ROOT:-.smoke_artifacts/fit_cleanup_full}"
BASELINE_DIR="${BASELINE_DIR:-$SMOKE_ROOT/baseline}"
CANDIDATE_DIR="${CANDIDATE_DIR:-$SMOKE_ROOT/candidate}"
SUMMARY_JSON="${SUMMARY_JSON:-$SMOKE_ROOT/compare_summary.json}"

GMM_FIT_SCRIPT="${GMM_FIT_SCRIPT:-fit_gmm.py}"
TUNED_LENS_FIT_SCRIPT="${TUNED_LENS_FIT_SCRIPT:-fit_tuned_lens.py}"
MODEL_NAME="${MODEL_NAME:-gpt2}"
REVISION="${REVISION:-main}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
LAYERS="${LAYERS:-all}"
TRACE_BATCH_SIZE="${TRACE_BATCH_SIZE:-100}"

GMM_LANGUAGES="${GMM_LANGUAGES:-fr tr}"
GMM_MAX_SAMPLES="${GMM_MAX_SAMPLES:-100}"
GMM_PCA_VARIANCE="${GMM_PCA_VARIANCE:-0.98}"

TUNED_LENS_LANGUAGES="${TUNED_LENS_LANGUAGES:-fr tr}"
# With the seeded 10% validation split, 18/lang yields 16/lang for one balanced batch of 32.
TUNED_LENS_MAX_SAMPLES_PER_LANG="${TUNED_LENS_MAX_SAMPLES_PER_LANG:-18}"
TUNED_LENS_BATCH_SIZE="${TUNED_LENS_BATCH_SIZE:-32}"
TUNED_LENS_MAX_TRAIN_STEPS="${TUNED_LENS_MAX_TRAIN_STEPS:-1}"
TUNED_LENS_MAX_LENGTH="${TUNED_LENS_MAX_LENGTH:-1024}"
TUNED_LENS_VAL_SPLIT="${TUNED_LENS_VAL_SPLIT:-0.1}"
TUNED_LENS_AMP_DTYPE="${TUNED_LENS_AMP_DTYPE:-bfloat16}"
TUNED_LENS_LAYERS_PER_STEP="${TUNED_LENS_LAYERS_PER_STEP:-1}"

RUN_GMM_TOKEN="${RUN_GMM_TOKEN:-1}"
RUN_GMM_SUBTOKEN="${RUN_GMM_SUBTOKEN:-1}"
RUN_TUNED_LENS="${RUN_TUNED_LENS:-1}"

# Set process-level determinism controls before Python and CUDA initialize.
export PYTHONHASHSEED="$SEED"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

model_safe() {
  python -c 'import re,sys; print(re.sub(r"[^A-Za-z0-9._-]+", "_", sys.argv[1].strip("/")))' "$1"
}

require_new_output_dir() {
  local output_dir="$1"
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to mix fitting artifacts into existing directory: $output_dir" >&2
    exit 2
  fi
  mkdir -p "$output_dir"
}

run_gmm_case() {
  local output_dir="$1"
  local repr_unit="$2"
  local -a layer_args=()
  local -a layer_indices=()
  read -r -a gmm_langs <<< "$GMM_LANGUAGES"
  if [[ -n "$LAYERS" && "$LAYERS" != "all" ]]; then
    read -r -a layer_indices <<< "$LAYERS"
    layer_args=(--layers "${layer_indices[@]}")
  fi

  python "$GMM_FIT_SCRIPT" \
    --model-name "$MODEL_NAME" \
    --revision "$REVISION" \
    --languages "${gmm_langs[@]}" \
    --data-root data/pud_holdout/pud_langs_train \
    --max-samples "$GMM_MAX_SAMPLES" \
    --device "$DEVICE" \
    --seed "$SEED" \
    "${layer_args[@]}" \
    --out-dir "$output_dir" \
    --repr-priors uniform \
    --repr-pca layerwise \
    --repr-pca-variance "$GMM_PCA_VARIANCE" \
    --repr-cov diag \
    --repr-unit "$repr_unit" \
    --trace-batch-size "$TRACE_BATCH_SIZE"
}

run_tuned_lens_case() {
  local output_dir="$1"
  local raw_root="$output_dir/.timestamped_run"
  local snapshot_parent
  local canonical_dir="$output_dir/tuned_lens"
  local -a layer_args=()
  local -a layer_indices=()
  local -a snapshots

  read -r -a tuned_lens_langs <<< "$TUNED_LENS_LANGUAGES"
  if [[ -n "$LAYERS" && "$LAYERS" != "all" ]]; then
    read -r -a layer_indices <<< "$LAYERS"
    layer_args=(--layers "${layer_indices[@]}")
  fi

  python "$TUNED_LENS_FIT_SCRIPT" \
    --model-name "$MODEL_NAME" \
    --revision "$REVISION" \
    --dataset-source pud \
    --languages "${tuned_lens_langs[@]}" \
    --max-samples-per-lang "$TUNED_LENS_MAX_SAMPLES_PER_LANG" \
    --device "$DEVICE" \
    --seed "$SEED" \
    "${layer_args[@]}" \
    --batch-size "$TUNED_LENS_BATCH_SIZE" \
    --num-epochs 1 \
    --max-train-steps "$TUNED_LENS_MAX_TRAIN_STEPS" \
    --val-split "$TUNED_LENS_VAL_SPLIT" \
    --light-val-examples-per-lang 2 \
    --val-every-steps 1 \
    --save-every-steps 1 \
    --early-stopping-patience 10 \
    --amp-dtype "$TUNED_LENS_AMP_DTYPE" \
    --max-length "$TUNED_LENS_MAX_LENGTH" \
    --layers-per-step "$TUNED_LENS_LAYERS_PER_STEP" \
    --out-dir "$raw_root"

  snapshot_parent="$raw_root/$(model_safe "$MODEL_NAME")/$REVISION"
  mapfile -t snapshots < <(find "$snapshot_parent" -mindepth 1 -maxdepth 1 -type d -print)
  if [[ "${#snapshots[@]}" -ne 1 ]]; then
    echo "Expected exactly one tuned-lens snapshot under $snapshot_parent; found ${#snapshots[@]}." >&2
    exit 2
  fi
  mv "${snapshots[0]}" "$canonical_dir"
}

run_phase() {
  local phase_name="$1"
  local output_dir="$2"
  require_new_output_dir "$output_dir"

  if [[ "$RUN_GMM_TOKEN" == "1" ]]; then
    run_gmm_case "$output_dir/gmm_token_layerwise_pca" token
  fi
  if [[ "$RUN_GMM_SUBTOKEN" == "1" ]]; then
    run_gmm_case "$output_dir/gmm_subtoken_layerwise_pca" subtoken
  fi
  if [[ "$RUN_TUNED_LENS" == "1" ]]; then
    run_tuned_lens_case "$output_dir"
  fi

  echo "Finished fitting smoke phase '$phase_name' in $output_dir"
}

compare_phases() {
  scripts/validation/compare_smoke_artifacts.py \
    "$BASELINE_DIR" \
    "$CANDIDATE_DIR" \
    --atol 0 \
    --rtol 0 \
    --summary-json "$SUMMARY_JSON"
}

case "$PHASE" in
  baseline)
    run_phase baseline "$BASELINE_DIR"
    ;;
  candidate)
    run_phase candidate "$CANDIDATE_DIR"
    ;;
  compare)
    compare_phases
    ;;
  both)
    run_phase baseline "$BASELINE_DIR"
    run_phase candidate "$CANDIDATE_DIR"
    compare_phases
    ;;
  *)
    echo "Unsupported PHASE=$PHASE; expected baseline, candidate, compare, or both." >&2
    exit 2
    ;;
esac
