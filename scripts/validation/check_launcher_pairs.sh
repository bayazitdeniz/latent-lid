#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

status=0
for runai_script in scripts/runai/runai_*.sh; do
  experiment="${runai_script##*/runai_}"
  run_script="scripts/runs/run_$experiment"
  # Exception: this is a scheduler-only script that reuses the standard local launcher.
  if [[ "$runai_script" == "scripts/runai/runai_include_lm_harness_waves.sh" ]]; then
    run_script="scripts/runs/run_include_lm_harness.sh"
  fi
  if [[ ! -f "$run_script" ]]; then
    echo "Missing local launcher for $runai_script: $run_script" >&2
    status=1
    continue
  fi
  if ! rg -Fq "$run_script" "$runai_script"; then
    echo "$runai_script does not source its canonical local launcher" >&2
    status=1
  fi
done

if rg -Fq 'submit_runai_job "$job_name" python "${CMD[@]}"' scripts/runai \
  || rg -Fq 'run_local_command python "${CMD[@]}"' scripts/runs scripts/tools; then
  echo "CMD arrays must include their command executable" >&2
  status=1
fi

if [[ "$status" != "0" ]]; then
  exit "$status"
fi

bash -n scripts/launcher_utils.sh scripts/data/*.sh scripts/runai/*.sh scripts/runs/*.sh scripts/tools/*.sh
echo "Launcher pairs and shell syntax are valid."
