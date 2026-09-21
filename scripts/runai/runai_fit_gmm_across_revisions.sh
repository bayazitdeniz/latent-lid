#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/runs/run_fit_gmm_across_revisions.sh"


################################################################################
# RunAI settings
################################################################################

JOB_PREFIX="${JOB_PREFIX:-gmm-train}"

################################################################################
# Submit the grid
################################################################################

for model in "${MODELS[@]}"; do
  load_revisions_for_model "$model"
  for revision in "${REVISIONS[@]}"; do
    for setup_name in "${GMM_SETUPS[@]}"; do
      resolved_batch="$(resolve_gmm_batch_size "$model" "$setup_name")"
      trace_batch_size="${resolved_batch%%$'\t'*}"
      batch_source="${resolved_batch#*$'\t'}"
      job_name="$(sanitize_job_name "$JOB_PREFIX-${setup_name//_/-}-$(model_short_job_name "$model")-$(revision_short_name "$model" "$revision")")"

      if [[ "$trace_batch_size" == "SKIP" ]]; then
        echo "Skipping $job_name: reason=$batch_source"
        continue
      fi

      build_fit_gmm_command "$model" "$setup_name" "$revision" "$trace_batch_size"
      echo "Submitting $job_name"
      echo "  model=$model"
      echo "  revision=$revision"
      echo "  setup=$setup_name"
      echo "  trace_batch_size=$trace_batch_size ($batch_source)"
      submit_runai_job "$job_name" "${CMD[@]}"
    done
  done
done
