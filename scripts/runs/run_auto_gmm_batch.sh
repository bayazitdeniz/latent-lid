#!/bin/bash
set -euo pipefail

# Local launcher for measuring safe GMM trace batch sizes.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/launcher_utils.sh


################################################################################
# Vars
################################################################################

REVISION="${REVISION:-main}"
OUT_JSON="${OUT_JSON:-logs/gmm_batch_sizes.json}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a GMM_SETUPS <<< "${GMM_SETUPS:-\
  pud9 \
  pud9_ud6 \
  pud21 \
  pud21_ud6 \
  include_10lang_en}"
read -r -a CANDIDATES <<< "${CANDIDATES:-1 2 4 8 16 32 64 100}"
read -r -a MODELS <<< "${MODELS:-\
  gpt2 \
  gpt2-xl \
  meta-llama/Llama-2-7b-hf \
  meta-llama/Llama-3.1-8B \
  meta-llama/Llama-3.1-8B-Instruct \
  swiss-ai/Apertus-8B-2509 \
  swiss-ai/Apertus-8B-Instruct-2509 \
  mistralai/Mistral-Nemo-Instruct-2407 \
  CohereLabs/aya-23-8B \
  utter-project/EuroLLM-9B \
  utter-project/EuroLLM-9B-Instruct \
  allenai/OLMo-2-1124-7B}"


################################################################################
# Funcs
################################################################################

gmm_batch_result_exists() {
  local model_name="$1"
  local setup_name="$2"
  python -c "import json, sys
from pathlib import Path
path = Path(sys.argv[1])
data = json.loads(path.read_text()) if path.exists() else {}
entry = data.get(sys.argv[2], {})
print('1' if isinstance(entry, dict) and sys.argv[3] in entry else '0')" \
    "$OUT_JSON" "$model_name" "$setup_name"
}

build_auto_gmm_command() {
  local model_name="$1"
  local setup_name="$2"
  CMD=(
    python
    -m scripts.tools.auto_gmm_batch_size
    --model-name "$model_name"
    --revision "$REVISION"
    --setup-name "$setup_name"
    --descending
    --out-json "$OUT_JSON"
  )
  CMD+=(--candidates "${CANDIDATES[@]}")
  return 0
}


################################################################################
# Run the grid
################################################################################

main() {
  local model
  local setup_name
  for model in "${MODELS[@]}"; do
    for setup_name in "${GMM_SETUPS[@]}"; do
      if [[ "$SKIP_EXISTING" == "1" && "$(gmm_batch_result_exists "$model" "$setup_name")" == "1" ]]; then
        echo "Skipping auto GMM batch probe: model=$model setup=$setup_name already in $OUT_JSON"
        continue
      fi
      build_auto_gmm_command "$model" "$setup_name"
      echo "Running auto GMM batch probe: model=$model setup=$setup_name"
      run_local_command "${CMD[@]}"
    done
  done
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
