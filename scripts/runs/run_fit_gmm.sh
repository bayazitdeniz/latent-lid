#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/launcher_utils.sh


################################################################################
# Vars
################################################################################

REVISION="${REVISION:-main}"
BATCH_SIZE_JSON="${BATCH_SIZE_JSON:-logs/gmm_batch_sizes.json}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a GMM_SETUPS <<< "${GMM_SETUPS:-\
  pud9 \
  pud9_ud6 \
  pud21 \
  pud21_ud6 \
  include_10lang_en}"
read -r -a MODELS <<< "${MODELS:-\
  gpt2 \
  gpt2-xl \
  meta-llama/Llama-2-7b-hf \
  meta-llama/Llama-3.1-8B \
  meta-llama/Llama-3.1-8B-Instruct \
  swiss-ai/Apertus-8B-2509 \
  swiss-ai/Apertus-8B-Instruct-2509 \
  mistralai/Mistral-Nemo-Instruct-2407 \
  CohereLabs/aya-23-8B \
  utter-project/EuroLLM-9B \
  utter-project/EuroLLM-9B-Instruct \
  allenai/OLMo-2-1124-7B}"


################################################################################
# Funcs
################################################################################

build_fit_gmm_command() {
  local model_name="$1"
  local setup_name="$2"
  local revision="$3"
  local trace_batch_size="$4"
  CMD=(
    python
    fit_gmm.py
    --model-name "$model_name"
    --revision "$revision"
    --setup-name "$setup_name"
    --trace-batch-size "$trace_batch_size"
  )
  [[ -n "${OUT_DIR:-}" ]] && CMD+=(--out-dir "$OUT_DIR")
  return 0
}

run_fit_gmm_job() {
  local model_name="$1"
  local setup_name="$2"
  local revision="$3"
  local resolved_batch
  local trace_batch_size
  local batch_source

  resolved_batch="$(resolve_gmm_batch_size "$model_name" "$setup_name")"
  trace_batch_size="${resolved_batch%%$'\t'*}"
  batch_source="${resolved_batch#*$'\t'}"
  if [[ "$trace_batch_size" == "SKIP" ]]; then
    echo "Skipping GMM fit: model=$model_name setup=$setup_name reason=$batch_source"
    SKIPPED_JOBS+=("setup=$setup_name | model=$model_name | reason=$batch_source")
    return
  fi

  build_fit_gmm_command "$model_name" "$setup_name" "$revision" "$trace_batch_size"
  echo "Running GMM fit: model=$model_name revision=$revision setup=$setup_name"
  echo "  trace_batch_size=$trace_batch_size ($batch_source)"
  RUN_COUNT=$((RUN_COUNT + 1))
  run_local_command "${CMD[@]}"
}


################################################################################
# Run the grid
################################################################################

main() {
  local model
  local setup_name
  RUN_COUNT=0
  SKIPPED_JOBS=()

  for model in "${MODELS[@]}"; do
    for setup_name in "${GMM_SETUPS[@]}"; do
      run_fit_gmm_job "$model" "$setup_name" "$REVISION"
    done
  done

  echo "Run summary: run=$RUN_COUNT skipped=${#SKIPPED_JOBS[@]}"
  if [[ "${#SKIPPED_JOBS[@]}" -gt 0 ]]; then
    printf '  - %s\n' "${SKIPPED_JOBS[@]}"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
