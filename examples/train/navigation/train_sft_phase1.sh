#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

SINGLE_STEP_FILE="${SINGLE_STEP_FILE:-/path/to/eb-nav_dataset_single_step.json}"
MULTI_STEP_FILE="${MULTI_STEP_FILE:-/path/to/eb-nav_dataset_multi_step.json}"
IMAGE_ROOT="${IMAGE_ROOT:-/path/to/EB-Nav_trajectory_dataset}"
HISTORY_MODE="${HISTORY_MODE:-no_history}"
INCLUDE_SOURCES="${INCLUDE_SOURCES:-both}"
ONLY_SUCCESSFUL_ACTIONS="${ONLY_SUCCESSFUL_ACTIONS:-false}"
SFT_DATA_OUTPUT="${SFT_DATA_OUTPUT:-/tmp/ebnav_phase1_${HISTORY_MODE}.parquet}"
MODEL_PATH="${MODEL_PATH:-$HOME/models/Qwen/Qwen2.5-VL-3B-Instruct}"
LORA_RANK="${LORA_RANK:-64}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-phase1_${HISTORY_MODE}_lora${LORA_RANK}}"

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

python -m vagen.envs.navigation.create_datasets.build_ebnav_sft "${builder_args[@]}"

python -m verl.trainer.sft_trainer \
  --config-path "$ROOT_DIR/vagen/configs" \
  --config-name navigation_sft \
  data.train_files="$SFT_DATA_OUTPUT" \
  data.builder.single_step_file="$SINGLE_STEP_FILE" \
  data.builder.multi_step_file="$MULTI_STEP_FILE" \
  data.builder.image_root="$IMAGE_ROOT" \
  data.builder.include_sources="$INCLUDE_SOURCES" \
  data.builder.history_mode="$HISTORY_MODE" \
  data.builder.only_successful_actions="$ONLY_SUCCESSFUL_ACTIONS" \
  model.path="$MODEL_PATH" \
  model.tokenizer_path="$MODEL_PATH" \
  model.lora_rank="$LORA_RANK" \
  trainer.experiment_name="$EXPERIMENT_NAME"
