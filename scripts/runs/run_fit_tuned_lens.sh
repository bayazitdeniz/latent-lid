#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/launcher_utils.sh


################################################################################
# Vars
################################################################################

REVISION="${REVISION:-main}"
BATCH_SIZE_JSON="${BATCH_SIZE_JSON:-logs/tuned_lens_batch_sizes.json}"
MAX_LENGTH="${MAX_LENGTH:-}"
WANDB="${WANDB:-1}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${DATASET_SOURCE:-fineweb}" == "pud" ]]; then
  BATCH_SIZE_DATASET_KEY="${BATCH_SIZE_DATASET_KEY:-pud}"
else
  BATCH_SIZE_DATASET_KEY="${BATCH_SIZE_DATASET_KEY:-fineweb27}"
fi

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
# Model-specific fitting settings
################################################################################

train_batch_size_for_model() {
  local model_name="$1"
  if [[ -n "${BATCH_SIZE:-}" ]]; then
    printf '%s\n' "$BATCH_SIZE"
    return
  fi
  if [[ -f "$BATCH_SIZE_JSON" ]]; then
    local resolved
    resolved="$(python -c "import json, sys
data = json.load(open(sys.argv[1]))
entry = data.get(sys.argv[2], {})
source = entry.get(sys.argv[3], {}) if isinstance(entry, dict) else {}
print(source.get('batch_size', ''))" \
      "$BATCH_SIZE_JSON" "$model_name" "$BATCH_SIZE_DATASET_KEY" 2>/dev/null || true)"
    if [[ -n "$resolved" ]]; then
      printf '%s\n' "$resolved"
      return
    fi
  fi
  case "$model_name" in
    gpt2|gpt2-xl) printf '4\n' ;;
    swiss-ai/Apertus-8B-Instruct-2509) printf '2\n' ;;
    *) printf '1\n' ;;
  esac
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


################################################################################
# Fitting commands
################################################################################

build_fit_tuned_lens_command() {
  local model_name="$1"
  local batch_size
  local max_length
  batch_size="$(train_batch_size_for_model "$model_name")"
  max_length="$(max_length_override_for_model "$model_name")"
  CMD=(
    python
    fit_tuned_lens.py
    --model-name "$model_name"
    --revision "$REVISION"
    --languages "${LANGUAGES[@]}"
  )

  [[ "$batch_size" != "1" ]] && CMD+=(--batch-size "$batch_size")
  [[ -n "$max_length" ]] && CMD+=(--max-length "$max_length")
  [[ -n "${DATASET_SOURCE:-}" ]] && CMD+=(--dataset-source "$DATASET_SOURCE")
  [[ -n "${FINEWEB_DATASET_DIR:-}" ]] && CMD+=(--fineweb-dataset-dir "$FINEWEB_DATASET_DIR")
  [[ -n "${MAX_SAMPLES_PER_LANG:-}" ]] && CMD+=(--max-samples-per-lang "$MAX_SAMPLES_PER_LANG")
  [[ -n "${DEVICE:-}" ]] && CMD+=(--device "$DEVICE")
  [[ -n "${OUT_DIR:-}" ]] && CMD+=(--out-dir "$OUT_DIR")

  if [[ -n "${LAYERS:-}" ]]; then
    local -a layer_array
    read -r -a layer_array <<< "$LAYERS"
    CMD+=(--layers "${layer_array[@]}")
  fi
  if [[ -n "${LIGHT_VAL_EXAMPLES_PER_LANG:-}" ]]; then
    CMD+=(--light-val-examples-per-lang "$LIGHT_VAL_EXAMPLES_PER_LANG")
  elif [[ "$model_name" == "CohereLabs/aya-23-8B" ]]; then
    CMD+=(--light-val-examples-per-lang 8)
  fi

  if [[ "$WANDB" == "1" ]]; then
    CMD+=(--wandb)
    [[ -n "${WANDB_PROJECT:-}" ]] && CMD+=(--wandb-project "$WANDB_PROJECT")
    [[ -n "${WANDB_ENTITY:-}" ]] && CMD+=(--wandb-entity "$WANDB_ENTITY")
    if [[ -n "${WANDB_TAGS:-}" ]]; then
      local -a tag_array
      read -r -a tag_array <<< "$WANDB_TAGS"
      CMD+=(--wandb-tags "${tag_array[@]}")
    fi
  fi
  return 0
}

run_fit_tuned_lens_job() {
  local model_name="$1"
  build_fit_tuned_lens_command "$model_name"
  echo "Running tuned-lens fit: model=$model_name revision=$REVISION"
  run_local_command "${CMD[@]}"
}


################################################################################
# Run the grid
################################################################################

main() {
  local model
  for model in "${MODELS[@]}"; do
    run_fit_tuned_lens_job "$model"
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
