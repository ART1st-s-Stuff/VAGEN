#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
if [[ -d "$ROOT_DIR/.venv/bin" ]]; then
  export PATH="$ROOT_DIR/.venv/bin:$PATH"
fi

DATA_ROOT="${DATA_ROOT:-$ROOT_DIR/datasets/navigation}"
SINGLE_STEP_FILE="${SINGLE_STEP_FILE:-$DATA_ROOT/eb-nav_dataset_single_step.json}"
MULTI_STEP_FILE="${MULTI_STEP_FILE:-$DATA_ROOT/eb-nav_dataset_multi_step.json}"
IMAGE_ROOT="${IMAGE_ROOT:-$DATA_ROOT/images}"
DATASET_REPO_ID="${DATASET_REPO_ID:-EmbodiedBench/EB-Nav_trajectory_dataset}"
HISTORY_MODE="${HISTORY_MODE:-no_history}"
INCLUDE_SOURCES="${INCLUDE_SOURCES:-both}"
ONLY_SUCCESSFUL_ACTIONS="${ONLY_SUCCESSFUL_ACTIONS:-false}"
SFT_DATA_OUTPUT="${SFT_DATA_OUTPUT:-$ROOT_DIR/datasets/phase1_debug/ebnav_phase1_${HISTORY_MODE}.parquet}"
MODEL_PATH="${MODEL_PATH:-$ROOT_DIR/models/Qwen2.5-VL-3B-Instruct}"
LORA_RANK="${LORA_RANK:-64}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-phase1_${HISTORY_MODE}_lora${LORA_RANK}}"
SINGLE_PROCESS="${SINGLE_PROCESS:-false}"
LAUNCHER="${LAUNCHER:-auto}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-false}"

# Keep YAML as source of truth by default.
# Only override these fields when env vars are explicitly provided.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-}"
MAX_LENGTH="${MAX_LENGTH:-}"
NUM_WORKERS="${NUM_WORKERS:-}"
PIN_MEMORY="${PIN_MEMORY:-}"

builder_args=(
  --output "$SFT_DATA_OUTPUT"
  --history_mode "$HISTORY_MODE"
  --include_sources "$INCLUDE_SOURCES"
  --single_step_file "$SINGLE_STEP_FILE"
  --multi_step_file "$MULTI_STEP_FILE"
  --image_root "$IMAGE_ROOT"
)

if [[ "$ONLY_SUCCESSFUL_ACTIONS" == "true" ]]; then
  builder_args+=(--only_successful_actions)
fi

mkdir -p "$(dirname "$SFT_DATA_OUTPUT")"
cd "$ROOT_DIR"

if [[ ! -f "$SINGLE_STEP_FILE" || ! -f "$MULTI_STEP_FILE" ]]; then
  echo "dataset json not found, downloading from ${DATASET_REPO_ID} ..."
  mkdir -p "$DATA_ROOT"
  DATASET_REPO_ID="$DATASET_REPO_ID" DATA_ROOT="$DATA_ROOT" python - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["DATASET_REPO_ID"],
    repo_type="dataset",
    local_dir=os.environ["DATA_ROOT"],
    local_dir_use_symlinks=False,
)
print(f"dataset ready: {os.environ['DATA_ROOT']}")
PY
fi

if [[ -f "$DATA_ROOT/images.zip" && ! -d "$DATA_ROOT/images" ]]; then
  echo "extracting images.zip ..."
  mkdir -p "$DATA_ROOT"
  unzip -oq "$DATA_ROOT/images.zip" -d "$DATA_ROOT"
fi

python -m vagen.envs.navigation.create_datasets.build_ebnav_sft "${builder_args[@]}"

if [[ "$LAUNCHER" == "auto" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LAUNCHER="srun"
  else
    LAUNCHER="torchrun"
  fi
fi

common_args=(
  -m verl.trainer.sft_trainer
  --config-path "$ROOT_DIR/vagen/configs"
  --config-name navigation_sft_phase1
  data.train_files="$SFT_DATA_OUTPUT"
  data.builder.single_step_file="$SINGLE_STEP_FILE"
  data.builder.multi_step_file="$MULTI_STEP_FILE"
  data.builder.image_root="$IMAGE_ROOT"
  data.builder.include_sources="$INCLUDE_SOURCES"
  data.builder.history_mode="$HISTORY_MODE"
  data.builder.only_successful_actions="$ONLY_SUCCESSFUL_ACTIONS"
  model.path="$MODEL_PATH"
  model.tokenizer_path="$MODEL_PATH"
  model.lora_rank="$LORA_RANK"
  trainer.experiment_name="$EXPERIMENT_NAME"
)

if [[ -n "$TRAIN_BATCH_SIZE" ]]; then
  common_args+=(data.train_batch_size="$TRAIN_BATCH_SIZE")
fi
if [[ -n "$MICRO_BATCH_SIZE" ]]; then
  common_args+=(data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE")
fi
if [[ -n "$MAX_LENGTH" ]]; then
  common_args+=(data.max_length="$MAX_LENGTH")
fi
if [[ -n "$NUM_WORKERS" ]]; then
  common_args+=(data.num_workers="$NUM_WORKERS")
fi
if [[ -n "$PIN_MEMORY" ]]; then
  common_args+=(data.pin_memory="$PIN_MEMORY")
fi

if [[ "$SINGLE_PROCESS" == "true" ]]; then
  cmd=(env WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR="$MASTER_ADDR" MASTER_PORT="$MASTER_PORT" python "${common_args[@]}")
elif [[ "$LAUNCHER" == "torchrun" ]]; then
  cmd=(python -m torch.distributed.run \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC_PER_NODE" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    "${common_args[@]}")
elif [[ "$LAUNCHER" == "srun" ]]; then
  cmd=(srun \
    --nodes="$NNODES" \
    --ntasks-per-node="$NPROC_PER_NODE" \
    $SRUN_EXTRA_ARGS \
    python "${common_args[@]}")
else
  echo "Unsupported LAUNCHER=$LAUNCHER (expected: auto|torchrun|srun)" >&2
  exit 1
fi

if [[ "$DRY_RUN" == "true" ]]; then
  printf 'launch command:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

"${cmd[@]}"
