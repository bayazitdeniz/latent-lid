#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/runs/run_include_lm_harness.sh"

################################################################################
# RunAI settings
################################################################################

JOB_PREFIX="${JOB_PREFIX:-llid-include-lmh}"

################################################################################
# Submit the grid
################################################################################

for task_group in "${TASK_GROUPS[@]}"; do
  for model in "${MODELS[@]}"; do
    job_name="$(sanitize_job_name "$JOB_PREFIX-$task_group-$(model_short_job_name "$model")")"
    build_include_lm_harness_command "$model" "$task_group"
    echo "Submitting $job_name"
    echo "  model=$model"
    echo "  tasks=$task_group"
    echo "  batch_size=auto:4"
    submit_runai_job "$job_name" "${CMD[@]}"
  done
done
