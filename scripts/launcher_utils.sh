#!/bin/bash

# Shared helpers for local and RunAI experiment launchers.
# This file is sourced; it is not an entrypoint itself.


################################################################################
# Names and paths
################################################################################

# Convert arbitrary text into a RunAI-safe job name.
sanitize_job_name() {
  tr '[:upper:]' '[:lower:]' <<< "$1" \
    | sed 's/[^[:alnum:]-]/-/g; s/-\{2,\}/-/g; s/^-//; s/-$//'
}

# Build the short model name used in jobs and as BPB/lm-harness artifact name.
model_short_job_name() {
  local model_name="$1"
  local model_part="${model_name##*/}"
  case "$model_name" in
    gpt2) model_part="gpt2" ;;
    gpt2-xl|openai-community/gpt2-xl) model_part="gpt2xl" ;;
    meta-llama/Llama-2-7b-hf) model_part="llama27b" ;;
    mistralai/Mistral-Nemo-Instruct-2407) model_part="nemo" ;;
    swiss-ai/Apertus-8B-2509) model_part="apertus8b" ;;
    swiss-ai/Apertus-8B-Instruct-2509) model_part="apertus8binstr" ;;
    allenai/OLMo-2-1124-7B) model_part="olmo211247b" ;;
  esac
  model_part="$(tr '[:upper:]' '[:lower:]' <<< "$model_part")"
  model_part="${model_part//instruct/instr}"
  tr -cd '[:alnum:]' <<< "$model_part"
}

# Convert a Hugging Face model name into its artifact-directory name.
model_artifact_name() {
  printf '%s\n' "${1//\//_}"
}

# Return the compact decoding-mode label used in job names.
short_decode_mode_name() {
  case "$1" in
    topk_weighted) printf 'topk\n' ;;
    rollout_argmax) printf 'argmax\n' ;;
    rollout_sample) printf 'topp\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}

# Return the compact lens label used in job names.
short_lens_name() {
  case "$1" in
    raw_logitlens) printf 'raw\n' ;;
    tuned_lens) printf 'tuned\n' ;;
    *) printf '%s\n' "$1" ;;
  esac
}


################################################################################
# Checkpoint/revision lists
################################################################################

# Load a model's token-aligned paper revisions into REVISIONS.
load_revisions_for_model() {
  local model_name="$1"
  local revision_file="checkpoint_lists/token_aligned_10/checkpoints_$(model_artifact_name "$model_name").txt"
  if [[ ! -f "$revision_file" ]]; then
    echo "Missing revisions file for $model_name: $revision_file" >&2
    return 1
  fi

  REVISIONS=()
  while IFS= read -r revision; do
    [[ -z "$revision" || "$revision" =~ ^[[:space:]]*# ]] && continue
    REVISIONS+=("$revision")
  done < "$revision_file"
  if [[ "${#REVISIONS[@]}" -eq 0 ]]; then
    echo "No revisions found in $revision_file" >&2
    return 1
  fi
  if [[ -n "${MAX_REVISIONS:-}" ]]; then
    if [[ ! "$MAX_REVISIONS" =~ ^[1-9][0-9]*$ ]]; then
      echo "MAX_REVISIONS must be a positive integer; got $MAX_REVISIONS" >&2
      return 1
    fi
    REVISIONS=("${REVISIONS[@]:0:MAX_REVISIONS}")
  fi
}

# Return the compact token or step label used for a revision.
revision_short_name() {
  local model_name="$1"
  local revision="$2"
  local alignment_file="checkpoint_lists/token_aligned_10/alignment.tsv"

  if [[ -f "$alignment_file" ]]; then
    local aligned_label
    aligned_label="$(python -c "import csv, sys
model, revision, path = sys.argv[1:4]
safe = model.replace('/', '_')
with open(path, newline='') as handle:
    for row in csv.DictReader(handle, delimiter='\\t'):
        if row.get(f'{safe}_revision') == revision:
            tokens = float(row[f'{safe}_tokens'])
            print(f'tok{int(round(tokens / 1e9))}b')
            break" "$model_name" "$revision" "$alignment_file" 2>/dev/null || true)"
    if [[ -n "$aligned_label" ]]; then
      printf '%s\n' "$aligned_label"
      return
    fi
  fi

  local token_part
  token_part="$(sed -n 's/.*tokens\([0-9][0-9.]*[kKmMbBtT]*\).*/\1/p' <<< "$revision" | tail -n 1)"
  if [[ -n "$token_part" ]]; then
    printf 'tok%s\n' "$(tr '[:upper:]' '[:lower:]' <<< "$token_part")"
    return
  fi

  local step_part
  step_part="$(sed -n 's/.*step\([0-9][0-9]*\).*/\1/p' <<< "$revision" | tail -n 1)"
  if [[ -n "$step_part" ]]; then
    printf 'step%s\n' "$step_part"
    return
  fi
  sanitize_job_name "$revision"
}


################################################################################
# Commands
################################################################################

# Print a shell-escaped command for logs and dry runs.
print_command() {
  printf '  command='
  printf '%q ' "$@"
  printf '\n'
}

# Print and time a local command unless it's a dry run.
run_local_command() {
  print_command "$@"
  if [[ "${DRY_RUN:-0}" != "1" ]]; then
    time "$@"
  fi
}

# Submit a RunAI job, or print its command if it's a dry run.
submit_runai_job() {
  local job_name="$1"
  local runai_image="${RUNAI_IMAGE:-}"
  local runai_pvc_mount="${RUNAI_PVC_MOUNT:-}"
  shift

  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    print_command "$@"
    return
  fi

  if [[ -z "$runai_image" ]]; then
    echo "RUNAI_IMAGE is required for RunAI submissions." >&2
    return 2
  fi
  if [[ -z "$runai_pvc_mount" ]]; then
    echo "RUNAI_PVC_MOUNT is required for RunAI submissions (format: <claim>:<container-path>)." >&2
    return 2
  fi
  if [[ "$runai_pvc_mount" != *:* ]]; then
    echo "Invalid RUNAI_PVC_MOUNT '$runai_pvc_mount'; expected <claim>:<container-path>." >&2
    return 2
  fi

  # GMM fitting and auto-batch jobs previously used 4 CPUs, 
  # so runtimes may differ slightly.
  runai submit \
    --name "$job_name" \
    -i "$runai_image" \
    --gpu 1 \
    --cpu 8 \
    --node-pool default \
    --pvc "$runai_pvc_mount" \
    --backoff-limit 0 \
    -- \
    "$@"
}


################################################################################
# GMM fitting batches
################################################################################

# Resolve a GMM trace batch size and report where it came from.
resolve_gmm_batch_size() {
  local model_name="$1"
  local setup_name="$2"
  if [[ -n "${TRACE_BATCH_SIZE:-}" ]]; then
    printf '%s\toverride\n' "$TRACE_BATCH_SIZE"
    return
  fi
  if [[ ! -f "$BATCH_SIZE_JSON" ]]; then
    printf '1\tfallback\n'
    return
  fi

  python -c "import json, sys
path, model, setup = sys.argv[1:4]
data = json.load(open(path))
entry = data.get(model)
batch = ''
source = 'auto'
if isinstance(entry, dict):
    setup_entry = entry.get(setup)
    fallbacks = {
        'include_10lang_en': 'pud21',
        'synthetic_anchor6_subtoken': 'pud21',
        'pud21_subtoken': 'pud21',
        'pud21_ud6_subtoken': 'pud21_ud6',
    }
    if setup_entry is None and setup in fallbacks:
        fallback = fallbacks[setup]
        setup_entry = entry.get(fallback)
        source = f'auto:{fallback}_fallback'
    if isinstance(setup_entry, dict):
        if setup_entry.get('failed') or setup_entry.get('batch_size') in (None, ''):
            print('SKIP\\tfailed_or_empty_auto_batch')
            sys.exit(0)
        batch = setup_entry.get('batch_size', '')
    elif 'batch_size' in entry:
        batch = entry.get('batch_size', '')
        source = 'legacy'
print(f'{batch}\\t{source}' if batch else '1\\tfallback')" \
    "$BATCH_SIZE_JSON" "$model_name" "$setup_name" 2>/dev/null \
    || printf '1\tfallback\n'
}
