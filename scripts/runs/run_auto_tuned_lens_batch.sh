#!/bin/bash
set -euo pipefail

# Local launcher for measuring safe tuned-lens training batch sizes.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/launcher_utils.sh


################################################################################
# Vars
################################################################################

REVISION="${REVISION:-main}"
BATCH_SIZE_DATASET_KEY="${BATCH_SIZE_DATASET_KEY:-fineweb27}"
OUT_JSON="${OUT_JSON:-logs/tuned_lens_batch_sizes.json}"
MAX_LENGTH="${MAX_LENGTH:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a LANGUAGES <<< "${LANGUAGES:-\
  ar bg cs de en es fa fi fr gl hi id is it ja ko mr pl pt_br ru sr sv th tr uk ur zh}"
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

tuned_lens_batch_result_exists() {
  local model_name="$1"
  local result_file="$2"
  python -c "import json, sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text()) if path.exists() else {}
entry = data.get(sys.argv[2], {})
print('1' if isinstance(entry, dict) and sys.argv[3] in entry else '0')" \
    "$result_file" "$model_name" "$BATCH_SIZE_DATASET_KEY"
}

max_length_override_for_model() {
  local model_name="$1"
  if [[ -n "$MAX_LENGTH" ]]; then
    printf '%s\n' "$MAX_LENGTH"
    return
  fi
  case "$model_name" in
    gpt2|gpt2-xl|CohereLabs/aya-23-8B|mistralai/Mistral-Nemo-Instruct-2407)
      printf '1024\n'
      ;;
    *) printf '\n' ;;
  esac
}

build_auto_tuned_lens_command() {
  local model_name="$1"
  local output_json="$2"
  local merge_json="${3:-}"
  local max_length
  max_length="$(max_length_override_for_model "$model_name")"
  CMD=(
    python
    -m scripts.tools.auto_tuned_lens_batch_size
    --model-name "$model_name"
    --revision "$REVISION"
    --batch-size-key "$BATCH_SIZE_DATASET_KEY"
    --languages "${LANGUAGES[@]}"
    --descending
    --out-json "$output_json"
  )
  [[ -n "$max_length" ]] && CMD+=(--max-length "$max_length")
  [[ -n "${FINEWEB_DATASET_DIR:-}" ]] && CMD+=(--fineweb-dataset-dir "$FINEWEB_DATASET_DIR")
  if [[ -n "${CANDIDATES:-}" ]]; then
    local -a candidate_array
    read -r -a candidate_array <<< "$CANDIDATES"
    CMD+=(--candidates "${candidate_array[@]}")
  fi
  [[ -n "$merge_json" ]] && CMD+=(--merge-into "$merge_json")
  return 0
}


################################################################################
# Run the grid
################################################################################

main() {
  local model
  for model in "${MODELS[@]}"; do
    if [[ "$SKIP_EXISTING" == "1" && "$(tuned_lens_batch_result_exists "$model" "$OUT_JSON")" == "1" ]]; then
      echo "Skipping tuned-lens batch probe: model=$model key=$BATCH_SIZE_DATASET_KEY already in $OUT_JSON"
      continue
    fi
    build_auto_tuned_lens_command "$model" "$OUT_JSON"
    echo "Running tuned-lens batch probe: model=$model"
    run_local_command "${CMD[@]}"
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
