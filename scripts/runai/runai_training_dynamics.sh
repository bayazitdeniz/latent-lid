#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/runs/run_training_dynamics.sh"

################################################################################
# RunAI settings
################################################################################

JOB_PREFIX="${JOB_PREFIX:-llid-train}"

################################################################################
# Submit the grid
################################################################################

for model in "${MODELS[@]}"; do
  load_revisions_for_model "$model"
  for revision in "${REVISIONS[@]}"; do
    if [[ "$RUN_REPR" == "1" ]]; then
      job_name="$(sanitize_job_name "$JOB_PREFIX-$(model_short_job_name "$model")-repr-$(revision_short_name "$model" "$revision")")"
      build_training_dynamics_command "$model" repr "$revision"
      echo "Submitting $job_name"
      echo "  model=$model"
      echo "  revision=$revision"
      echo "  mode=repr"
      submit_runai_job "$job_name" "${CMD[@]}"
    fi
    if [[ "$RUN_DECODING" == "1" ]]; then
      for decoding_mode in "${DECODING_MODES[@]}"; do
        method_name="dec-$(short_decode_mode_name "$decoding_mode")-raw"
        job_name="$(sanitize_job_name "$JOB_PREFIX-$(model_short_job_name "$model")-$method_name-$(revision_short_name "$model" "$revision")")"
        build_training_dynamics_command "$model" decoding "$revision" "$decoding_mode"
        echo "Submitting $job_name"
        echo "  model=$model"
        echo "  revision=$revision"
        echo "  mode=decoding"
        submit_runai_job "$job_name" "${CMD[@]}"
      done
    fi
  done
done
