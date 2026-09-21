#!/bin/bash
set -euo pipefail

if [[ ! -f run_eval.py || ! -d scripts ]]; then
  echo "Run this script from the repository root." >&2
  exit 2
fi

PHASE="${PHASE:-both}" # baseline | candidate | compare | both
SMOKE_ROOT="${SMOKE_ROOT:-.smoke_artifacts/cleanup_smoke}"
BASELINE_DIR="${BASELINE_DIR:-$SMOKE_ROOT/baseline}"
CANDIDATE_DIR="${CANDIDATE_DIR:-$SMOKE_ROOT/candidate}"
SUMMARY_JSON="${SUMMARY_JSON:-$SMOKE_ROOT/compare_summary.json}"

SMOKE_MODELS="${SMOKE_MODELS:-swiss-ai/Apertus-8B-2509}"
REVISION="${REVISION:-main}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
LAYERS="${LAYERS:-0 1 2}"
TRACE_BATCH_SIZE="${TRACE_BATCH_SIZE:-1}"
MAX_PROMPTS_PER_LANG="${MAX_PROMPTS_PER_LANG:-1}"
MAX_PROMPTS="${MAX_PROMPTS:-8}"

PUD_ROOT="${PUD_ROOT:-data/pud_holdout}"
PUD_SPLIT_MODE="${PUD_SPLIT_MODE:-heldout}"
SYNTHETIC_ROOT="${SYNTHETIC_ROOT:-data/wendler_2024_data/common69}"
TARGET_STRING_DATA_SOURCE="${TARGET_STRING_DATA_SOURCE:-translation}"

RUN_TARGET_STRING="${RUN_TARGET_STRING:-1}"
RUN_DECODE_ARGMAX="${RUN_DECODE_ARGMAX:-1}"
RUN_DECODE_SAMPLE="${RUN_DECODE_SAMPLE:-1}"
RUN_REPR="${RUN_REPR:-1}"

DECODING_LID_BACKEND="${DECODING_LID_BACKEND:-langid}"
DECODING_TOKEN_AGG="${DECODING_TOKEN_AGG:-frac_50}"
DECODING_ROLLOUT_K="${DECODING_ROLLOUT_K:-1}"
DECODING_ROLLOUT_WORD_CNT="${DECODING_ROLLOUT_WORD_CNT:-0}"
DECODING_ROLLOUT_TOP_P="${DECODING_ROLLOUT_TOP_P:-0.9}"
DECODING_ROLLOUT_NUM_SAMPLES="${DECODING_ROLLOUT_NUM_SAMPLES:-2}"
SAMPLED_ROLLOUT_PROMPT_BATCH_SIZE="${SAMPLED_ROLLOUT_PROMPT_BATCH_SIZE:-1}"

REPR_GMM_DIR="${REPR_GMM_DIR:-}"
REPR_GMM_DIR_TEMPLATE="${REPR_GMM_DIR_TEMPLATE:-}"
REPR_DATA_SOURCE="${REPR_DATA_SOURCE:-pud21}"
REPR_GMM_SETUP="${REPR_GMM_SETUP:-}"
REPR_TOKEN_AGGS="${REPR_TOKEN_AGGS:-last_token frac_50}"

model_safe() {
  python -c 'import re,sys; print(re.sub(r"[^A-Za-z0-9._-]+", "_", sys.argv[1].strip("/")))' "$1"
}

tokenizer_artifact_safe() {
  python -c 'import sys; from data.wendler_2024_data.target_string_artifacts import safe_tokenizer_artifact_name; print(safe_tokenizer_artifact_name(sys.argv[1]))' "$1"
}

target_string_start_token_path() {
  local model="$1"
  local safe
  safe="$(tokenizer_artifact_safe "$model")"
  echo "$SYNTHETIC_ROOT/target_string_start_tokens/$safe/${TARGET_STRING_DATA_SOURCE}_start_tokens.jsonl"
}

check_target_string_inputs() {
  if [[ "$RUN_TARGET_STRING" != "1" ]]; then
    return
  fi

  local missing=0
  for model in $SMOKE_MODELS; do
    local path
    path="$(target_string_start_token_path "$model")"
    if [[ ! -f "$path" ]]; then
      echo "Missing target-string Start(w) artifact for model '$model': $path" >&2
      missing=1
    fi
  done

  if [[ "$missing" == "1" ]]; then
    echo "" >&2
    echo "The cleanup smoke target-string case intentionally uses Start(w) scoring." >&2
    echo "Set SYNTHETIC_ROOT to a data directory with tokenizer-specific start-token artifacts, or run without this case:" >&2
    echo "  RUN_TARGET_STRING=0 PHASE=$PHASE bash scripts/validation/run_cleanup_smoke_matrix.sh" >&2
    exit 2
  fi
}

check_phase_inputs() {
  check_target_string_inputs
}

run_eval_case() {
  local output_dir="$1"
  local exp_id="$2"
  shift 2
  python run_eval.py \
    --revision "$REVISION" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --layers $LAYERS \
    --trace-batch-size "$TRACE_BATCH_SIZE" \
    --sampled-rollout-prompt-batch-size "$SAMPLED_ROLLOUT_PROMPT_BATCH_SIZE" \
    --output-dir "$output_dir" \
    --exp-id "$exp_id" \
    "$@"
}

repr_dir_for_model() {
  local model="$1"
  local safe
  safe="$(model_safe "$model")"
  if [[ -n "$REPR_GMM_DIR_TEMPLATE" ]]; then
    local dir="$REPR_GMM_DIR_TEMPLATE"
    dir="${dir//\{model\}/$model}"
    dir="${dir//\{model_safe\}/$safe}"
    dir="${dir//\{revision\}/$REVISION}"
    echo "$dir"
  else
    echo "$REPR_GMM_DIR"
  fi
}

run_phase() {
  local phase_name="$1"
  local output_dir="$2"
  mkdir -p "$output_dir"

  for model in $SMOKE_MODELS; do
    local safe
    safe="$(model_safe "$model")"

    if [[ "$RUN_TARGET_STRING" == "1" ]]; then
      run_eval_case "$output_dir" "target_string_${safe}" \
        --model-name "$model" \
        --data-source "$TARGET_STRING_DATA_SOURCE" \
        --synthetic-root "$SYNTHETIC_ROOT" \
        --max-prompts "$MAX_PROMPTS" \
        --max-prompts-per-lang 0 \
        --do-decoding true \
        --do-repr false \
        --decoding-lens raw_logitlens \
        --decoding-mapping target_string \
        --target-string-scoring-mode start_tokens_only \
        --decoding-token-agg last_token \
        --write-prompt-metrics \
        --save-full-lang-probs
    fi

    if [[ "$RUN_DECODE_ARGMAX" == "1" ]]; then
      run_eval_case "$output_dir" "decode_argmax_${safe}" \
        --model-name "$model" \
        --data-source pud21 \
        --pud-root "$PUD_ROOT" \
        --pud-split-mode "$PUD_SPLIT_MODE" \
        --max-prompts 0 \
        --max-prompts-per-lang "$MAX_PROMPTS_PER_LANG" \
        --do-decoding true \
        --do-repr false \
        --decoding-lens raw_logitlens \
        --decoding-mapping decode_then_classify \
        --decoding-decode-mode rollout_argmax \
        --decoding-token-agg "$DECODING_TOKEN_AGG" \
        --decoding-rollout-k "$DECODING_ROLLOUT_K" \
        --decoding-rollout-word-cnt "$DECODING_ROLLOUT_WORD_CNT" \
        --decoding-lid-backend "$DECODING_LID_BACKEND" \
        --write-prompt-metrics \
        --save-full-lang-probs
    fi

    if [[ "$RUN_DECODE_SAMPLE" == "1" ]]; then
      run_eval_case "$output_dir" "decode_sample_${safe}" \
        --model-name "$model" \
        --data-source pud21 \
        --pud-root "$PUD_ROOT" \
        --pud-split-mode "$PUD_SPLIT_MODE" \
        --max-prompts 0 \
        --max-prompts-per-lang "$MAX_PROMPTS_PER_LANG" \
        --do-decoding true \
        --do-repr false \
        --decoding-lens raw_logitlens \
        --decoding-mapping decode_then_classify \
        --decoding-decode-mode rollout_sample \
        --decoding-token-agg "$DECODING_TOKEN_AGG" \
        --decoding-rollout-k "$DECODING_ROLLOUT_K" \
        --decoding-rollout-word-cnt "$DECODING_ROLLOUT_WORD_CNT" \
        --decoding-rollout-top-p "$DECODING_ROLLOUT_TOP_P" \
        --decoding-rollout-num-samples "$DECODING_ROLLOUT_NUM_SAMPLES" \
        --decoding-lid-backend "$DECODING_LID_BACKEND" \
        --write-prompt-metrics \
        --save-full-lang-probs
    fi

    if [[ "$RUN_REPR" == "1" ]]; then
      local repr_dir
      repr_dir="$(repr_dir_for_model "$model")"
      for repr_token_agg in $REPR_TOKEN_AGGS; do
        local -a cmd_args
        cmd_args=(
          --model-name "$model" \
          --data-source "$REPR_DATA_SOURCE" \
          --pud-root "$PUD_ROOT" \
          --pud-split-mode "$PUD_SPLIT_MODE" \
          --max-prompts 0 \
          --max-prompts-per-lang "$MAX_PROMPTS_PER_LANG" \
          --do-decoding false \
          --do-repr true \
          --repr-token-agg "$repr_token_agg" \
          --repr-rollout-k 0 \
          --write-prompt-metrics \
          --save-full-lang-probs
        )
        if [[ -n "$repr_dir" ]]; then
          cmd_args+=(--repr-gmm-dir "$repr_dir")
        elif [[ -n "$REPR_GMM_SETUP" ]]; then
          cmd_args+=(--repr-gmm-setup "$REPR_GMM_SETUP")
        fi
        run_eval_case "$output_dir" "repr_${repr_token_agg}_${safe}" "${cmd_args[@]}"
      done
    fi
  done

  echo "Finished smoke phase '$phase_name' in $output_dir"
}

compare_phases() {
  scripts/validation/compare_smoke_artifacts.py \
    "$BASELINE_DIR" \
    "$CANDIDATE_DIR" \
    --atol 0 \
    --rtol 0 \
    --ignore-json-key exp_id \
    --ignore-json-key output_dir \
    --summary-json "$SUMMARY_JSON"
}

case "$PHASE" in
  baseline)
    check_phase_inputs
    run_phase baseline "$BASELINE_DIR"
    ;;
  candidate)
    check_phase_inputs
    run_phase candidate "$CANDIDATE_DIR"
    ;;
  compare)
    compare_phases
    ;;
  both)
    check_phase_inputs
    run_phase baseline "$BASELINE_DIR"
    run_phase candidate "$CANDIDATE_DIR"
    compare_phases
    ;;
  *)
    echo "Unsupported PHASE=$PHASE; expected baseline, candidate, compare, or both." >&2
    exit 2
    ;;
esac
