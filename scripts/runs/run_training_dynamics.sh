#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/launcher_utils.sh

################################################################################
# Vars
################################################################################

DATA_SOURCE="${DATA_SOURCE:-pud21}"
RUN_REPR="${RUN_REPR:-1}"
RUN_DECODING="${RUN_DECODING:-1}"
MAX_PROMPTS="${MAX_PROMPTS:-}"
MAX_PROMPTS_PER_LANG="${MAX_PROMPTS_PER_LANG:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a MODELS <<< "${MODELS:-\
  swiss-ai/Apertus-8B-2509 \
  allenai/OLMo-2-1124-7B}"
read -r -a DECODING_MODES \
  <<< "${DECODING_DECODE_MODE:-${DECODING_MODES:-rollout_argmax rollout_sample}}"


################################################################################
# Funcs
################################################################################

build_training_dynamics_command() {
  local model_name="$1"
  local mode="$2"
  local revision="$3"
  local decoding_mode="${4:-}"
  CMD=(
    python
    run_eval.py
    --model-name "$model_name"
    --data-source "$DATA_SOURCE"
  )

  [[ "$revision" != "main" ]] && CMD+=(--revision "$revision")
  [[ -n "$MAX_PROMPTS" ]] && CMD+=(--max-prompts "$MAX_PROMPTS")
  [[ -n "$MAX_PROMPTS_PER_LANG" ]] && CMD+=(--max-prompts-per-lang "$MAX_PROMPTS_PER_LANG")
  [[ -n "$OUTPUT_DIR" ]] && CMD+=(--output-dir "$OUTPUT_DIR")

  if [[ "$mode" == "repr" ]]; then
    CMD+=(--do-decoding false --do-repr true)
    return
  fi
  if [[ "$mode" != "decoding" ]]; then
    echo "Unknown evaluation mode: $mode" >&2
    return 1
  fi

  [[ "$decoding_mode" != "rollout_sample" ]] && CMD+=(--decoding-decode-mode "$decoding_mode")
  return 0
}

run_training_dynamics_job() {
  build_training_dynamics_command "$@"
  echo "Running training-dynamics evaluation: model=$1 mode=$2 revision=$3"
  run_local_command "${CMD[@]}"
}


################################################################################
# Run the grid
################################################################################

main() {
  local model
  local revision
  local decoding_mode
  for model in "${MODELS[@]}"; do
    load_revisions_for_model "$model"
    for revision in "${REVISIONS[@]}"; do
      [[ "$RUN_REPR" == "1" ]] \
        && run_training_dynamics_job "$model" repr "$revision"
      if [[ "$RUN_DECODING" == "1" ]]; then
        for decoding_mode in "${DECODING_MODES[@]}"; do
          run_training_dynamics_job "$model" decoding "$revision" "$decoding_mode"
        done
      fi
    done
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
