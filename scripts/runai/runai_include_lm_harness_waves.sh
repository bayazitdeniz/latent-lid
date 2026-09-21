#!/bin/bash
set -euo pipefail

# Throttled wrapper around runai_include_lm_harness.sh.
# This avoids cold-starting many INCLUDE jobs at once, which can trigger
# Hugging Face dataset API rate limits while lm-eval resolves 44 languages.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TASK_GROUPS="${TASK_GROUPS:-include_base_44 include_base_44_few_shot_en include_base_44_few_shot_og}"
source "$REPO_ROOT/scripts/runs/run_include_lm_harness.sh"

################################################################################
# RunAI settings
################################################################################

JOB_PREFIX_BASE="${JOB_PREFIX_BASE:-inc-lmh}"
WAVE_SIZE="${WAVE_SIZE:-3}"
# sleep for 15 minutes between waves to avoid rate limits
SLEEP_SECONDS="${SLEEP_SECONDS:-900}"

task_job_prefix() {
  case "$1" in
    include_base_44) printf '%s0\n' "$JOB_PREFIX_BASE" ;;
    include_base_44_few_shot_en) printf '%s5en\n' "$JOB_PREFIX_BASE" ;;
    include_base_44_few_shot_og) printf '%s5og\n' "$JOB_PREFIX_BASE" ;;
    *) printf '%s\n' "$JOB_PREFIX_BASE" ;;
  esac
}

################################################################################
# Submit the grid in throttled waves
################################################################################

for task_index in "${!TASK_GROUPS[@]}"; do
  task_group="${TASK_GROUPS[$task_index]}"
  echo "Submitting INCLUDE lm-harness task group: $task_group"
  for ((start = 0; start < ${#MODELS[@]}; start += WAVE_SIZE)); do
    wave=("${MODELS[@]:start:WAVE_SIZE}")
    for model in "${wave[@]}"; do
      job_name="$(sanitize_job_name "$(task_job_prefix "$task_group")-$(model_short_job_name "$model")")"
      build_include_lm_harness_command "$model" "$task_group"
      echo "Submitting $job_name"
      echo "  model=$model"
      echo "  tasks=$task_group"
      echo "  batch_size=auto:4"
      submit_runai_job "$job_name" "${CMD[@]}"
    done

    next=$((start + WAVE_SIZE))
    if [[ "$DRY_RUN" != "1" && "$next" -lt "${#MODELS[@]}" ]]; then
      echo "Sleeping ${SLEEP_SECONDS}s before the next wave"
      sleep "$SLEEP_SECONDS"
    fi
  done

  if [[ "$DRY_RUN" != "1" && "$task_index" -lt $((${#TASK_GROUPS[@]} - 1)) ]]; then
    echo "Sleeping ${SLEEP_SECONDS}s before the next task group"
    sleep "$SLEEP_SECONDS"
  fi
done
