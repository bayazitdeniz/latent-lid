#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/runs/run_auto_gmm_batch.sh"

################################################################################
# RunAI settings
################################################################################

JOB_PREFIX="${JOB_PREFIX:-gmm-batch}"

################################################################################
# Submit the grid
################################################################################

for model in "${MODELS[@]}"; do
  for setup_name in "${GMM_SETUPS[@]}"; do
    if [[ "$SKIP_EXISTING" == "1" && "$(gmm_batch_result_exists "$model" "$setup_name")" == "1" ]]; then
      echo "Skipping auto GMM batch probe: model=$model setup=$setup_name already in $OUT_JSON"
      continue
    fi
    job_name="$(sanitize_job_name "$JOB_PREFIX-${setup_name//_/-}-$(model_short_job_name "$model")-$REVISION")"
    build_auto_gmm_command "$model" "$setup_name"
    echo "Submitting $job_name"
    echo "  model=$model"
    echo "  revision=$REVISION"
    echo "  setup=$setup_name"
    submit_runai_job "$job_name" "${CMD[@]}"
  done
done
