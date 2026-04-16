#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
SCRIPT_DIR="${SCRIPT_DIR:-scripts/test}"
if [[ -d "$REPO_ROOT/.venv/bin" ]]; then
  export PATH="$REPO_ROOT/.venv/bin:$PATH"
fi

MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/Qwen2.5-VL-3B-Instruct}"
TRAIN_FILE="${TRAIN_FILE:-$REPO_ROOT/datasets/phase1_debug/ebnav_small.json}"
DATA_ROOT="${DATA_ROOT:-$REPO_ROOT/datasets/navigation}"
SINGLE_STEP_FILE="${SINGLE_STEP_FILE:-$DATA_ROOT/eb-nav_dataset_single_step.json}"
MULTI_STEP_FILE="${MULTI_STEP_FILE:-$DATA_ROOT/eb-nav_dataset_multi_step.json}"
IMAGE_ROOT="${IMAGE_ROOT:-$DATA_ROOT}"
HISTORY_MODE="${HISTORY_MODE:-no_history}"
INCLUDE_SOURCES="${INCLUDE_SOURCES:-single_step}"
ONLY_SUCCESSFUL_ACTIONS="${ONLY_SUCCESSFUL_ACTIONS:-false}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-phase1_smoke_${HISTORY_MODE}}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/models/phase1_smoke/$EXPERIMENT_NAME}"

TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
LORA_RANK="${LORA_RANK:-64}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PIN_MEMORY="${PIN_MEMORY:-true}"
SINGLE_PROCESS="${SINGLE_PROCESS:-true}"
LAUNCHER="${LAUNCHER:-auto}"
NNODES="${NNODES:-1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
SRUN_EXTRA_ARGS="${SRUN_EXTRA_ARGS:-}"
DRY_RUN="${DRY_RUN:-false}"

if [[ ! -d "$MODEL_PATH" ]]; then
  MODEL_DIR="$MODEL_PATH" "$SCRIPT_DIR/init_phase1_model.sh"
fi

if [[ ! -f "$TRAIN_FILE" ]]; then
  OUTPUT_FILE="$TRAIN_FILE" \
  HISTORY_MODE="$HISTORY_MODE" \
  INCLUDE_SOURCES="$INCLUDE_SOURCES" \
  ONLY_SUCCESSFUL_ACTIONS="$ONLY_SUCCESSFUL_ACTIONS" \
  SINGLE_STEP_FILE="$SINGLE_STEP_FILE" \
  MULTI_STEP_FILE="$MULTI_STEP_FILE" \
  IMAGE_ROOT="$IMAGE_ROOT" \
  "$SCRIPT_DIR/init_phase1_data.sh"
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/verl:$PYTHONPATH"
else
  export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/verl"
fi

mkdir -p "$OUTPUT_DIR"
cd "$REPO_ROOT"

if [[ "$LAUNCHER" == "auto" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    LAUNCHER="srun"
  else
    LAUNCHER="torchrun"
  fi
fi

common_args=(
  -m verl.trainer.sft_trainer
  --config-path "$REPO_ROOT/vagen/configs"
  --config-name navigation_sft_phase1
  data.train_files="$TRAIN_FILE"
  data.train_max_samples="$TRAIN_MAX_SAMPLES"
  data.train_batch_size="$TRAIN_BATCH_SIZE"
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE"
  data.max_length="$MAX_LENGTH"
  data.num_workers="$NUM_WORKERS"
  data.pin_memory="$PIN_MEMORY"
  data.builder.single_step_file="$SINGLE_STEP_FILE"
  data.builder.multi_step_file="$MULTI_STEP_FILE"
  data.builder.image_root="$IMAGE_ROOT"
  data.builder.include_sources="$INCLUDE_SOURCES"
  data.builder.history_mode="$HISTORY_MODE"
  data.builder.only_successful_actions="$ONLY_SUCCESSFUL_ACTIONS"
  model.path="$MODEL_PATH"
  model.tokenizer_path="$MODEL_PATH"
  model.lora_rank="$LORA_RANK"
  trainer.project_name=navigation-phase1-sft
  trainer.experiment_name="$EXPERIMENT_NAME"
  trainer.default_local_dir="$OUTPUT_DIR"
  trainer.logger="['console']"
  trainer.total_epochs=1
  trainer.save_freq=-1
  trainer.test_freq=-1
  trainer.resume_mode=disable
)

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
