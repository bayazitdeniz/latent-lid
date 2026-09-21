#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/runs/run_auto_tuned_lens_batch.sh"

MERGE_JSON="${MERGE_JSON:-$OUT_JSON}"
OUT_JSON_DIR="${OUT_JSON_DIR:-logs/tuned_lens_batch_sizes_fragments}"

################################################################################
# RunAI settings
################################################################################

JOB_PREFIX="${JOB_PREFIX:-tlens-auto}"

################################################################################
# Submit the grid
################################################################################

for model in "${MODELS[@]}"; do
  if [[ "$SKIP_EXISTING" == "1" && "$(tuned_lens_batch_result_exists "$model" "$MERGE_JSON")" == "1" ]]; then
    echo "Skipping tuned-lens batch probe: model=$model key=$BATCH_SIZE_DATASET_KEY already in $MERGE_JSON"
    continue
  fi
  fragment_path="$OUT_JSON_DIR/$(model_artifact_name "$model").json"
  job_name="$(sanitize_job_name "$JOB_PREFIX-$(model_short_job_name "$model")")"
  build_auto_tuned_lens_command "$model" "$fragment_path" "$MERGE_JSON"
  echo "Submitting $job_name"
  echo "  model=$model"
  echo "  revision=$REVISION"
  submit_runai_job "$job_name" "${CMD[@]}"
done
