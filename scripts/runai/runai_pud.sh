#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/runs/run_pud.sh"


################################################################################
# RunAI settings
################################################################################

JOB_PREFIX="${JOB_PREFIX:-llid}"

################################################################################
# Submit the grid
################################################################################

submit_pud_grid() {
  local data_source="$1"
  local mode="$2"
  local decoding_lens="${3:-}"
  local decoding_mode="${4:-}"
  local model
  local method_name="$mode"

  if [[ "$mode" == "decoding" ]]; then
    method_name="dec-$(short_decode_mode_name "$decoding_mode")-$(short_lens_name "$decoding_lens")"
  fi
  for model in "${MODELS[@]}"; do
    job_name="$(sanitize_job_name "$JOB_PREFIX-$data_source-$(model_short_job_name "$model")-$method_name")"
    build_pud_command "$model" "$data_source" "$mode" "$decoding_lens" "$decoding_mode"
    echo "Submitting $job_name"
    echo "  model=$model"
    echo "  data_source=$data_source"
    echo "  mode=$mode"
    submit_runai_job "$job_name" "${CMD[@]}"
  done
}

for data_source in "${DATA_SOURCES[@]}"; do
  [[ "$LAUNCH_REPR" == "1" ]] && submit_pud_grid "$data_source" repr
  if [[ "$LAUNCH_RAW_LOGITLENS" == "1" ]]; then
    [[ "$LAUNCH_ROLLOUT_ARGMAX" == "1" ]] \
      && submit_pud_grid "$data_source" decoding raw_logitlens rollout_argmax
    [[ "$LAUNCH_ROLLOUT_SAMPLE" == "1" ]] \
      && submit_pud_grid "$data_source" decoding raw_logitlens rollout_sample
  fi
  if [[ "$LAUNCH_TUNED_LENS" == "1" ]] && uses_tuned_lens "$data_source"; then
    [[ "$LAUNCH_ROLLOUT_ARGMAX" == "1" ]] \
      && submit_pud_grid "$data_source" decoding tuned_lens rollout_argmax
    [[ "$LAUNCH_ROLLOUT_SAMPLE" == "1" ]] \
      && submit_pud_grid "$data_source" decoding tuned_lens rollout_sample
  fi
done
