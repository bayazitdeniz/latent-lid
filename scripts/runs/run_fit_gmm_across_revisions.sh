#!/bin/bash
set -euo pipefail

# Use the same fitting command as the final-checkpoint launcher, but over the
# token-aligned Apertus/OLMo checkpoint lists used in the paper.
GMM_SETUPS="${GMM_SETUPS:-pud21}"
MODELS="${MODELS:-swiss-ai/Apertus-8B-2509 allenai/OLMo-2-1124-7B}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO_ROOT/scripts/runs/run_fit_gmm.sh"


################################################################################
# Run the grid
################################################################################

main() {
  local model
  local revision
  local setup_name
  RUN_COUNT=0
  SKIPPED_JOBS=()

  for model in "${MODELS[@]}"; do
    load_revisions_for_model "$model"
    for revision in "${REVISIONS[@]}"; do
      for setup_name in "${GMM_SETUPS[@]}"; do
        run_fit_gmm_job "$model" "$setup_name" "$revision"
      done
    done
  done

  echo "Run summary: run=$RUN_COUNT skipped=${#SKIPPED_JOBS[@]}"
  if [[ "${#SKIPPED_JOBS[@]}" -gt 0 ]]; then
    printf '  - %s\n' "${SKIPPED_JOBS[@]}"
  fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
