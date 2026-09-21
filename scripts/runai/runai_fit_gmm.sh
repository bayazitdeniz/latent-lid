#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/runs/run_fit_gmm.sh"


################################################################################
# RunAI settings
################################################################################

JOB_PREFIX="${JOB_PREFIX:-gmm}"

################################################################################
# Submit the grid
################################################################################

SUBMITTED_COUNT=0
SKIPPED_JOBS=()
for model in "${MODELS[@]}"; do
  for setup_name in "${GMM_SETUPS[@]}"; do
    resolved_batch="$(resolve_gmm_batch_size "$model" "$setup_name")"
    trace_batch_size="${resolved_batch%%$'\t'*}"
    batch_source="${resolved_batch#*$'\t'}"

    name_parts="$JOB_PREFIX"
    if [[ "${#GMM_SETUPS[@]}" -gt 1 ]]; then
      name_parts+="-${setup_name//_/-}"
    fi
    job_name="$(sanitize_job_name "$name_parts-$(model_short_job_name "$model")")"
    if [[ "$REVISION" != "main" ]]; then
      job_name="$(sanitize_job_name "$job_name-$REVISION")"
    fi

    if [[ "$trace_batch_size" == "SKIP" ]]; then
      echo "Skipping $job_name: reason=$batch_source"
      SKIPPED_JOBS+=("setup=$setup_name | model=$model | reason=$batch_source")
      continue
    fi

    build_fit_gmm_command "$model" "$setup_name" "$REVISION" "$trace_batch_size"
    echo "Submitting $job_name"
    echo "  model=$model"
    echo "  revision=$REVISION"
    echo "  setup=$setup_name"
    echo "  trace_batch_size=$trace_batch_size ($batch_source)"
    SUBMITTED_COUNT=$((SUBMITTED_COUNT + 1))
    submit_runai_job "$job_name" "${CMD[@]}"
  done
done

echo "Submission summary: submitted=$SUBMITTED_COUNT skipped=${#SKIPPED_JOBS[@]}"
if [[ "${#SKIPPED_JOBS[@]}" -gt 0 ]]; then
  printf '  - %s\n' "${SKIPPED_JOBS[@]}"
fi
