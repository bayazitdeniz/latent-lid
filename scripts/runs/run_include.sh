#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/launcher_utils.sh


################################################################################
# Paper grid
################################################################################

LAUNCH_REPR="${LAUNCH_REPR:-1}"
LAUNCH_ROLLOUT_ARGMAX="${LAUNCH_ROLLOUT_ARGMAX:-1}"
LAUNCH_ROLLOUT_SAMPLE="${LAUNCH_ROLLOUT_SAMPLE:-1}"
LAUNCH_RAW_LOGITLENS="${LAUNCH_RAW_LOGITLENS:-1}"
LAUNCH_TUNED_LENS="${LAUNCH_TUNED_LENS:-1}"
MAX_PROMPTS="${MAX_PROMPTS:-}"
MAX_PROMPTS_PER_LANG="${MAX_PROMPTS_PER_LANG:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a DATA_SOURCES \
  <<< "${DATA_SOURCES:-${DATA_SOURCE:-include_10lang_3domain_cap30}}"

# GPT-2 and GPT-2 XL truncate these long reading-comprehension prompts.
read -r -a MODELS <<< "${MODELS:-\
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
# Evaluation commands
################################################################################

short_include_data_source() {
  case "$1" in
    include_10lang_3domain_cap30) printf 'inc-cap30\n' ;;
    include_10lang_3domain_all) printf 'inc-all\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

build_include_command() {
  local model_name="$1"
  local data_source="$2"
  local mode="$3"
  local decoding_lens="${4:-}"
  local decoding_mode="${5:-}"
  CMD=(
    python
    run_eval.py
    --model-name "$model_name"
    --data-source "$data_source"
  )

  [[ -n "$MAX_PROMPTS" ]] && CMD+=(--max-prompts "$MAX_PROMPTS")
  [[ -n "$MAX_PROMPTS_PER_LANG" ]] && CMD+=(--max-prompts-per-lang "$MAX_PROMPTS_PER_LANG")
  [[ -n "$OUTPUT_DIR" ]] && CMD+=(--output-dir "$OUTPUT_DIR")

  # representation args
  if [[ "$mode" == "repr" ]]; then
    CMD+=(--do-decoding false --do-repr true)
    [[ -n "${REPR_TOKEN_AGG:-}" ]] && CMD+=(--repr-token-agg "$REPR_TOKEN_AGG")
    [[ -n "${REPR_ROLLOUT_K:-}" ]] && CMD+=(--repr-rollout-k "$REPR_ROLLOUT_K")
    [[ -n "${REPR_UNIT:-}" ]] && CMD+=(--repr-unit "$REPR_UNIT")
    return 0
  fi

  if [[ "$mode" != "decoding" ]]; then
    echo "Unknown evaluation mode: $mode" >&2
    return 1
  fi

  # decoding args
  [[ "$decoding_lens" != "raw_logitlens" ]] && CMD+=(--decoding-lens "$decoding_lens")
  [[ "$decoding_mode" != "rollout_sample" ]] && CMD+=(--decoding-decode-mode "$decoding_mode")
  [[ -n "${DECODING_TOKEN_AGG:-}" ]] && CMD+=(--decoding-token-agg "$DECODING_TOKEN_AGG")
  [[ -n "${DECODING_LID_BACKEND:-}" ]] && CMD+=(--decoding-lid-backend "$DECODING_LID_BACKEND")
  [[ -n "${DECODING_TOPK:-}" ]] && CMD+=(--decoding-topk "$DECODING_TOPK")
  [[ -n "${DECODING_ROLLOUT_K:-}" ]] && CMD+=(--decoding-rollout-k "$DECODING_ROLLOUT_K")
  [[ -n "${DECODING_ROLLOUT_WORD_CNT:-}" ]] && CMD+=(--decoding-rollout-word-cnt "$DECODING_ROLLOUT_WORD_CNT")
  [[ -n "${DECODING_ROLLOUT_TOP_P:-}" ]] && CMD+=(--decoding-rollout-top-p "$DECODING_ROLLOUT_TOP_P")
  [[ -n "${DECODING_ROLLOUT_NUM_SAMPLES:-}" ]] && CMD+=(--decoding-rollout-num-samples "$DECODING_ROLLOUT_NUM_SAMPLES")
  [[ -n "${TRACE_BATCH_SIZE:-}" ]] && CMD+=(--trace-batch-size "$TRACE_BATCH_SIZE")
  [[ -n "${SAMPLED_ROLLOUT_PROMPT_BATCH_SIZE:-}" ]] \
    && CMD+=(--sampled-rollout-prompt-batch-size "$SAMPLED_ROLLOUT_PROMPT_BATCH_SIZE")
  return 0
}

run_include_job() {
  build_include_command "$@"
  echo "Running INCLUDE evaluation: model=$1 data_source=$2 mode=$3"
  run_local_command "${CMD[@]}"
}

run_include_method_grid() {
  local data_source="$1"
  local mode="$2"
  local decoding_lens="${3:-}"
  local decoding_mode="${4:-}"
  local model
  for model in "${MODELS[@]}"; do
    run_include_job "$model" "$data_source" "$mode" "$decoding_lens" "$decoding_mode"
  done
}


################################################################################
# Run the grid
################################################################################

main() {
  local data_source
  for data_source in "${DATA_SOURCES[@]}"; do
    [[ "$LAUNCH_REPR" == "1" ]] && run_include_method_grid "$data_source" repr
    if [[ "$LAUNCH_RAW_LOGITLENS" == "1" ]]; then
      [[ "$LAUNCH_ROLLOUT_ARGMAX" == "1" ]] \
        && run_include_method_grid "$data_source" decoding raw_logitlens rollout_argmax
      [[ "$LAUNCH_ROLLOUT_SAMPLE" == "1" ]] \
        && run_include_method_grid "$data_source" decoding raw_logitlens rollout_sample
    fi
    if [[ "$LAUNCH_TUNED_LENS" == "1" ]]; then
      [[ "$LAUNCH_ROLLOUT_ARGMAX" == "1" ]] \
        && run_include_method_grid "$data_source" decoding tuned_lens rollout_argmax
      [[ "$LAUNCH_ROLLOUT_SAMPLE" == "1" ]] \
        && run_include_method_grid "$data_source" decoding tuned_lens rollout_sample
    fi
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
