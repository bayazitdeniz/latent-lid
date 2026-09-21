#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/launcher_utils.sh


################################################################################
# Paper grid
################################################################################

TRANSLATION_MODELS="${TRANSLATION_MODELS:-}"
COPY_MODELS="${COPY_MODELS:-}"
CLOZE_MODELS="${CLOZE_MODELS:-}"
EXP_PREFIX="${EXP_PREFIX:-${JOB_PREFIX:-llid-ts}}"
MAX_PROMPTS="${MAX_PROMPTS:-}"
MAX_PROMPTS_PER_LANG="${MAX_PROMPTS_PER_LANG:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TRACE_BATCH_SIZE="${TRACE_BATCH_SIZE:-}"
RUN_REPR="${RUN_REPR:-0}"
RUN_DECODING="${RUN_DECODING:-1}"
REPR_GMM_SETUP="${REPR_GMM_SETUP:-pud21_ud6_subtoken}"
REPR_TOKEN_AGG="${REPR_TOKEN_AGG:-last_token}"
REPR_ROLLOUT_K="${REPR_ROLLOUT_K:-0}"
REPR_UNIT="${REPR_UNIT:-subtoken}"
SYNTHETIC_LANGS="${SYNTHETIC_LANGS:-}"
DECODING_LENS="${DECODING_LENS:-raw_logitlens}"
HARD_EXIT_AFTER_SUCCESS="${HARD_EXIT_AFTER_SUCCESS:-1}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a DATA_SOURCES <<< "${DATA_SOURCES:-\
  copy \
  cloze \
  translation_to_ar \
  translation_to_en \
  translation_to_fr \
  translation_to_hi \
  translation_to_ru \
  translation_to_tr \
  translation_to_zh}"
read -r -a MODELS <<< "${MODELS:-\
  meta-llama/Llama-2-7b-hf \
  meta-llama/Llama-3.1-8B \
  meta-llama/Llama-3.1-8B-Instruct \
  swiss-ai/Apertus-8B-2509 \
  swiss-ai/Apertus-8B-Instruct-2509 \
  mistralai/Mistral-Nemo-Instruct-2407 \
  CohereLabs/aya-23-8B \
  utter-project/EuroLLM-9B \
  utter-project/EuroLLM-9B-Instruct}"


################################################################################
# Evaluation commands
################################################################################

models_for_data_source() {
  local data_source="$1"
  local override=""
  case "$data_source" in
    translation|translation_to_*) override="$TRANSLATION_MODELS" ;;
    copy) override="$COPY_MODELS" ;;
    cloze) override="$CLOZE_MODELS" ;;
  esac

  if [[ -n "$override" && "${override,,}" != "all" ]]; then
    read -r -a SELECTED_MODELS <<< "$override"
  else
    SELECTED_MODELS=("${MODELS[@]}")
  fi
}

controlled_exp_id() {
  local model_name="$1"
  local data_source="$2"
  sanitize_job_name "$EXP_PREFIX-${data_source//_/-}-$(model_short_job_name "$model_name")"
}

build_controlled_command() {
  local model_name="$1"
  local data_source="$2"

  CMD=(
    python
    run_eval.py
    --model-name "$model_name"
    --data-source "$data_source"
    --exp-id "$(controlled_exp_id "$model_name" "$data_source")"
  )
  
  [[ -n "${SYNTHETIC_ROOT:-}" ]] && CMD+=(--synthetic-root "$SYNTHETIC_ROOT")
  [[ "$RUN_DECODING" != "1" ]] && CMD+=(--do-decoding false)
  [[ "$RUN_REPR" == "1" ]] && CMD+=(--do-repr true)

  if [[ -n "$SYNTHETIC_LANGS" ]]; then
    local -a synthetic_language_array
    read -r -a synthetic_language_array <<< "$SYNTHETIC_LANGS"
    CMD+=(--synthetic-langs "${synthetic_language_array[@]}")
  fi

  [[ -n "$MAX_PROMPTS" ]] && CMD+=(--max-prompts "$MAX_PROMPTS")
  [[ -n "$MAX_PROMPTS_PER_LANG" ]] && CMD+=(--max-prompts-per-lang "$MAX_PROMPTS_PER_LANG")
  [[ -n "$OUTPUT_DIR" ]] && CMD+=(--output-dir "$OUTPUT_DIR")
  [[ -n "$TRACE_BATCH_SIZE" ]] && CMD+=(--trace-batch-size "$TRACE_BATCH_SIZE")

  if [[ "$RUN_REPR" == "1" ]]; then
    CMD+=(--repr-token-agg "$REPR_TOKEN_AGG" --repr-rollout-k "$REPR_ROLLOUT_K")
    [[ "$REPR_UNIT" != "token" ]] && CMD+=(--repr-unit "$REPR_UNIT")
    [[ -n "$REPR_GMM_SETUP" ]] && CMD+=(--repr-gmm-setup "$REPR_GMM_SETUP")
  fi

  if [[ "$RUN_DECODING" == "1" ]]; then
    CMD+=(--decoding-token-agg last_token)
    [[ "$DECODING_LENS" != "raw_logitlens" ]] && CMD+=(--decoding-lens "$DECODING_LENS")
  fi

  [[ "$HARD_EXIT_AFTER_SUCCESS" == "1" ]] && CMD+=(--hard-exit-after-success)
  return 0
}

run_controlled_job() {
  local model_name="$1"
  local data_source="$2"
  build_controlled_command "$model_name" "$data_source"
  echo "Running controlled evaluation: model=$model_name data_source=$data_source"
  run_local_command "${CMD[@]}"
}


################################################################################
# Run the grid
################################################################################

main() {
  local data_source
  local model
  if [[ "$RUN_DECODING" == "$RUN_REPR" ]]; then
    echo "Set exactly one of RUN_DECODING and RUN_REPR to 1." >&2
    return 1
  fi
  for data_source in "${DATA_SOURCES[@]}"; do
    case "$data_source" in
      translation|translation_to_*|copy|cloze) ;;
      *)
        echo "Unsupported DATA_SOURCES entry: $data_source" >&2
        return 1
        ;;
    esac
    models_for_data_source "$data_source"
    for model in "${SELECTED_MODELS[@]}"; do
      run_controlled_job "$model" "$data_source"
    done
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
