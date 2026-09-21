#!/bin/bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
source scripts/launcher_utils.sh

OUTPUT_ROOT="${OUTPUT_ROOT:-logs/bpb}"
MAX_EXAMPLES_PER_LANG="${MAX_EXAMPLES_PER_LANG:-}"
DRY_RUN="${DRY_RUN:-0}"

read -r -a MODELS <<< "${MODELS:-\
  gpt2 \
  openai-community/gpt2-xl \
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

for model in "${MODELS[@]}"; do
  output_name="$(model_short_job_name "$model")"
  CMD=(
    python
    scripts/tools/compute_bpb.py
    --model-name "$model"
    --output-csv "$OUTPUT_ROOT/${output_name}_pud21_ud6.csv"
    --output-jsonl "$OUTPUT_ROOT/${output_name}_pud21_ud6.jsonl"
    --line-output-csv "$OUTPUT_ROOT/${output_name}_pud21_ud6_lines.csv"
  )
  [[ -n "$MAX_EXAMPLES_PER_LANG" ]] \
    && CMD+=(--max-examples-per-lang "$MAX_EXAMPLES_PER_LANG")
  echo "Running PUD BPB evaluation: model=$model"
  run_local_command "${CMD[@]}"
done
