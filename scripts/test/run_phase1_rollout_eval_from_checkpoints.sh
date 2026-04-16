#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -d "$REPO_ROOT/.venv/bin" ]]; then
  export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/verl:$PYTHONPATH"
else
  export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/verl"
fi

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$REPO_ROOT/checkpoints/navigation-phase1-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
STEP_DIR="${STEP_DIR:-}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$REPO_ROOT/models/Qwen2.5-VL-3B-Instruct}"

EXPORT_ROOT="${EXPORT_ROOT:-$REPO_ROOT/outputs/phase1_merged_models}"
FORCE_REMERGE="${FORCE_REMERGE:-false}"

SINGLE_STEP_FILE="${SINGLE_STEP_FILE:-$REPO_ROOT/datasets/navigation/eb-nav_dataset_single_step.json}"
MULTI_STEP_FILE="${MULTI_STEP_FILE:-$REPO_ROOT/datasets/navigation/eb-nav_dataset_multi_step.json}"
INCLUDE_SOURCES="${INCLUDE_SOURCES:-single_step}"
IMAGE_ROOT="${IMAGE_ROOT:-$REPO_ROOT/datasets/navigation}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/phase1_rollout_eval}"
SAVE_TESTSET="${SAVE_TESTSET:-true}"
DRY_RUN="${DRY_RUN:-false}"

has_hf_model_weights() {
  local p="$1"
  [[ -f "$p/model.safetensors" || -f "$p/model.safetensors.index.json" || -f "$p/pytorch_model.bin" || -f "$p/pytorch_model.bin.index.json" ]] && return 0
  compgen -G "$p/model-*.safetensors" >/dev/null && return 0
  return 1
}

has_lora_adapter() {
  local p="$1"
  [[ -f "$p/adapter_config.json" ]] || return 1
  [[ -f "$p/adapter_model.safetensors" || -f "$p/adapter_model.bin" ]] && return 0
  compgen -G "$p/adapter_model-*.safetensors" >/dev/null && return 0
  return 1
}

resolve_step_dir() {
  if [[ -n "$STEP_DIR" ]]; then
    if [[ ! -d "$STEP_DIR" ]]; then
      echo "STEP_DIR does not exist: $STEP_DIR" >&2
      return 1
    fi
    echo "$STEP_DIR"
    return 0
  fi

  local exp_dir=""
  if [[ -n "$EXPERIMENT_NAME" ]]; then
    exp_dir="$CHECKPOINT_ROOT/$EXPERIMENT_NAME"
    if [[ ! -d "$exp_dir" ]]; then
      echo "EXPERIMENT_NAME does not exist: $exp_dir" >&2
      return 1
    fi
  else
    exp_dir="$(ls -dt "$CHECKPOINT_ROOT"/*/ 2>/dev/null | head -n 1 || true)"
    exp_dir="${exp_dir%/}"
    if [[ -z "$exp_dir" ]]; then
      echo "No experiment found under $CHECKPOINT_ROOT" >&2
      return 1
    fi
  fi

  local latest_iter_file="$exp_dir/latest_checkpointed_iteration.txt"
  if [[ -f "$latest_iter_file" ]]; then
    local step
    step="$(tr -d '[:space:]' < "$latest_iter_file")"
    if [[ -n "$step" && -d "$exp_dir/global_step_$step" ]]; then
      echo "$exp_dir/global_step_$step"
      return 0
    fi
  fi

  local latest_step_dir
  latest_step_dir="$(ls -dt "$exp_dir"/global_step_*/ 2>/dev/null | head -n 1 || true)"
  latest_step_dir="${latest_step_dir%/}"
  if [[ -n "$latest_step_dir" ]]; then
    echo "$latest_step_dir"
    return 0
  fi

  echo "No global_step checkpoint found under $exp_dir" >&2
  return 1
}

resolve_export_dir() {
  local resolved_step_dir="$1"
  local step_name
  step_name="$(basename "$resolved_step_dir")"
  local exp_name
  exp_name="$(basename "$(dirname "$resolved_step_dir")")"
  echo "$EXPORT_ROOT/$exp_name/$step_name"
}

merge_checkpoint_to_hf() {
  local resolved_step_dir="$1"
  local export_dir="$2"

  if [[ "$FORCE_REMERGE" != "true" ]]; then
    if has_hf_model_weights "$export_dir"; then
      echo "Reuse merged HF model: $export_dir"
      return 0
    fi
  fi

  mkdir -p "$export_dir"
  echo "Merging FSDP checkpoint to HF format..."
  python "$REPO_ROOT/verl/scripts/legacy_model_merger.py" merge \
    --backend fsdp \
    --local_dir "$resolved_step_dir" \
    --target_dir "$export_dir"
}

cd "$REPO_ROOT"
RESOLVED_STEP_DIR="$(resolve_step_dir)"
EXPORT_DIR="$(resolve_export_dir "$RESOLVED_STEP_DIR")"

if [[ "$DRY_RUN" == "true" ]]; then
  echo "Resolved step checkpoint: $RESOLVED_STEP_DIR"
  echo "Export directory: $EXPORT_DIR"
  echo "Dry run: skip merge and eval."
  exit 0
fi

merge_checkpoint_to_hf "$RESOLVED_STEP_DIR" "$EXPORT_DIR"

MODEL_PATH=""
EXTRA_ARGS=()
if has_lora_adapter "$EXPORT_DIR/lora_adapter"; then
  MODEL_PATH="$EXPORT_DIR/lora_adapter"
  EXTRA_ARGS+=(--base-model-path "$BASE_MODEL_PATH")
  echo "Detected LoRA adapter export: $MODEL_PATH"
  echo "Using base model: $BASE_MODEL_PATH"
elif has_hf_model_weights "$EXPORT_DIR"; then
  MODEL_PATH="$EXPORT_DIR"
  echo "Detected merged full model: $MODEL_PATH"
else
  echo "Merged output missing expected files under $EXPORT_DIR" >&2
  exit 1
fi

cmd=(
  python "$REPO_ROOT/scripts/test/phase1_rollout_testset_and_eval.py"
  --single-step-file "$SINGLE_STEP_FILE"
  --multi-step-file "$MULTI_STEP_FILE"
  --include-sources "$INCLUDE_SOURCES"
  --image-root "$IMAGE_ROOT"
  --num-rollouts "$NUM_ROLLOUTS"
  --model-path "$MODEL_PATH"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --batch-size "$BATCH_SIZE"
  --device "$DEVICE"
  --dtype "$DTYPE"
  --output-dir "$OUTPUT_DIR"
)

if [[ "$SAVE_TESTSET" == "true" ]]; then
  cmd+=(--save-testset)
fi
cmd+=("${EXTRA_ARGS[@]}")

printf 'launch command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
