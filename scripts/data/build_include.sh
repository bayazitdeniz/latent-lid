#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/launcher_utils.sh

DRY_RUN="${DRY_RUN:-0}"

echo "Building the capped INCLUDE subset"
run_local_command python -m scripts.data.build_include_data \
  --out-dir data/include_10lang_3domain_cap30 \
  --sampling cap \
  --max-per-cell 30

echo "Building the uncapped INCLUDE subset"
run_local_command python -m scripts.data.build_include_data \
  --out-dir data/include_10lang_3domain_all
