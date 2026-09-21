#!/bin/bash
set -euo pipefail

# This launcher uses data from lm-evaluation-harness, not the materialized
# INCLUDE subset used by run_eval.py.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/launcher_utils.sh

################################################################################
# Paper grid
################################################################################

TASK_GROUPS="${TASK_GROUPS:-include_base_44}"
OUTPUT_ROOT="${OUTPUT_ROOT:-logs/lm_harness/include}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a MODELS <<< "${MODELS:-\
  gpt2 \
  openai-community/gpt2-xl \
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
read -r -a TASK_GROUPS <<< "$TASK_GROUPS"

INCLUDE_BASE_44_LANGUAGES=(
  albanian arabic armenian azerbaijani basque belarusian bengali bulgarian
  chinese croatian dutch estonian finnish french georgian german greek hebrew
  hindi hungarian indonesian italian japanese kazakh korean lithuanian malay
  malayalam nepali "north macedonian" persian polish portuguese russian serbian
  spanish tagalog tamil telugu turkish ukrainian urdu uzbek vietnamese
)


################################################################################
# lm-evaluation-harness commands
################################################################################

join_by_comma() {
  local IFS=,
  printf '%s\n' "$*"
}

include_base_44_tasks() {
  local suffix="$1"
  local -a tasks=()
  local language
  for language in "${INCLUDE_BASE_44_LANGUAGES[@]}"; do
    tasks+=("include_base_44_${language}${suffix}")
  done
  join_by_comma "${tasks[@]}"
}

effective_tasks() {
  case "$1" in
    include_base_44) include_base_44_tasks "" ;;
    include_base_44_few_shot_en) include_base_44_tasks "_few_shot_en" ;;
    include_base_44_few_shot_og) include_base_44_tasks "_few_shot_og" ;;
    *) printf '%s\n' "$1" ;;
  esac
}

effective_num_fewshot() {
  local task_group="$1"
  if [[ "$task_group" == *_few_shot_en || "$task_group" == *_few_shot_og ]]; then
    printf '5\n'
  else
    printf '0\n'
  fi
}

build_model_args() {
  local model_name="$1"
  printf 'pretrained=%s,trust_remote_code=True\n' "$model_name"
}

build_include_lm_harness_command() {
  local model_name="$1"
  local task_group="$2"
  CMD=(
    python
    -m
    lm_eval
    --model hf
    --model_args "$(build_model_args "$model_name")"
    --tasks "$(effective_tasks "$task_group")"
    --num_fewshot "$(effective_num_fewshot "$task_group")"
    --batch_size auto:4
    --output_path "$OUTPUT_ROOT/$task_group/$(model_short_job_name "$model_name")"
  )
  [[ -n "$LIMIT" ]] && CMD+=(--limit "$LIMIT")
  return 0
}

run_include_lm_harness_job() {
  local model_name="$1"
  local task_group="$2"
  build_include_lm_harness_command "$model_name" "$task_group"
  echo "Running INCLUDE lm-harness: model=$model_name tasks=$task_group"
  run_local_command "${CMD[@]}"
}


################################################################################
# Run the grid
################################################################################

main() {
  local task_group
  local model
  for task_group in "${TASK_GROUPS[@]}"; do
    for model in "${MODELS[@]}"; do
      run_include_lm_harness_job "$model" "$task_group"
    done
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
