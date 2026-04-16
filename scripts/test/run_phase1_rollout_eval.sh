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
MODEL_PATH="${MODEL_PATH:-}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$REPO_ROOT/models/Qwen2.5-VL-3B-Instruct}"

SINGLE_STEP_FILE="${SINGLE_STEP_FILE:-$REPO_ROOT/datasets/navigation/eb-nav_dataset_single_step.json}"
MULTI_STEP_FILE="${MULTI_STEP_FILE:-$REPO_ROOT/datasets/navigation/eb-nav_dataset_multi_step.json}"
INCLUDE_SOURCES="${INCLUDE_SOURCES:-single_step}"
IMAGE_ROOT="${IMAGE_ROOT:-$REPO_ROOT/datasets/navigation}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DEVICE="${DEVICE:-auto}"
DTYPE="${DTYPE:-auto}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/phase1_rollout_eval}"
SAVE_TESTSET="${SAVE_TESTSET:-true}"
DRY_RUN="${DRY_RUN:-false}"

resolve_model_path() {
  if [[ -n "$MODEL_PATH" ]]; then
    echo "$MODEL_PATH"
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

MODEL_PATH="$(resolve_model_path)"

if [[ -d "$MODEL_PATH/huggingface" && -f "$MODEL_PATH/huggingface/config.json" ]]; then
  # Prefer exported HF subdir when available.
  MODEL_PATH="$MODEL_PATH/huggingface"
fi

IS_LORA="false"
if has_hf_model_weights "$MODEL_PATH"; then
  IS_LORA="false"
elif has_lora_adapter "$MODEL_PATH"; then
  IS_LORA="true"
else
  echo "Model weights not found under: $MODEL_PATH" >&2
  echo "Expected one of:" >&2
  echo "  - full model: model.safetensors / model-*.safetensors / pytorch_model.bin (+index)" >&2
  echo "  - LoRA adapter: adapter_config.json + adapter_model.safetensors" >&2
  echo "Hint: your current checkpoint seems to be FSDP shards; export adapter/full HF weights first." >&2
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
  --device "$DEVICE"
  --dtype "$DTYPE"
  --output-dir "$OUTPUT_DIR"
)

if [[ "$IS_LORA" == "true" ]]; then
  cmd+=(--base-model-path "$BASE_MODEL_PATH")
fi

if [[ "$SAVE_TESTSET" == "true" ]]; then
  cmd+=(--save-testset)
fi

cd "$REPO_ROOT"

echo "Using model checkpoint: $MODEL_PATH"
if [[ "$IS_LORA" == "true" ]]; then
  echo "Detected LoRA adapter mode, base model: $BASE_MODEL_PATH"
fi
if [[ "$DRY_RUN" == "true" ]]; then
  printf 'launch command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
