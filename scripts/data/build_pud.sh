#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/launcher_utils.sh

DRY_RUN="${DRY_RUN:-0}"

# The paper artifacts use the canonical UD tarball rather than the HF mirror.
echo "Building the PUD train/test split"
run_local_command python -m scripts.data.build_pud_data
