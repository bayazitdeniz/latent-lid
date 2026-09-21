#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/runs/run_fit_tuned_lens.sh"

JOB_PREFIX="${JOB_PREFIX:-llid-tlens}"


################################################################################
# Submit the grid
################################################################################

for model in "${MODELS[@]}"; do
  job_name="$(sanitize_job_name "$JOB_PREFIX-$(model_short_job_name "$model")-$REVISION")"
  build_fit_tuned_lens_command "$model"
  echo "Submitting $job_name"
  echo "  model=$model"
  echo "  revision=$REVISION"
  submit_runai_job "$job_name" "${CMD[@]}"
done
