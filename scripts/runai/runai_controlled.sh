#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/runs/run_controlled.sh"

################################################################################
# RunAI settings
################################################################################

JOB_PREFIX="${JOB_PREFIX:-llid-ts}"

################################################################################
# Submit the grid
################################################################################

if [[ "$RUN_DECODING" == "$RUN_REPR" ]]; then
  echo "Set exactly one of RUN_DECODING and RUN_REPR to 1." >&2
  exit 1
fi
for data_source in "${DATA_SOURCES[@]}"; do
  case "$data_source" in
    translation|translation_to_*|copy|cloze) ;;
    *)
      echo "Unsupported DATA_SOURCES entry: $data_source" >&2
      exit 1
      ;;
  esac
  models_for_data_source "$data_source"
  for model in "${SELECTED_MODELS[@]}"; do
    job_name="$(sanitize_job_name "$JOB_PREFIX-${data_source//_/-}-$(model_short_job_name "$model")")"
    build_controlled_command "$model" "$data_source"
    echo "Submitting $job_name"
    echo "  model=$model"
    echo "  data_source=$data_source"
    submit_runai_job "$job_name" "${CMD[@]}"
  done
done
