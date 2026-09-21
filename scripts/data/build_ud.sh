#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/launcher_utils.sh

DRY_RUN="${DRY_RUN:-0}"

echo "Building the non-parallel UD train/test split"
run_local_command python -m scripts.data.build_ud_data
