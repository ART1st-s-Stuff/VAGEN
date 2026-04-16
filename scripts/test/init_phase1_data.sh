#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-.}"

DATA_ROOT="${DATA_ROOT:-datasets/navigation}"
SINGLE_STEP_FILE="${SINGLE_STEP_FILE:-$DATA_ROOT/eb-nav_dataset_single_step.json}"
MULTI_STEP_FILE="${MULTI_STEP_FILE:-$DATA_ROOT/eb-nav_dataset_multi_step.json}"
IMAGE_ROOT="${IMAGE_ROOT:-$DATA_ROOT/images}"
DATASET_REPO_ID="${DATASET_REPO_ID:-EmbodiedBench/EB-Nav_trajectory_dataset}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/datasets/phase1_debug}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/ebnav_small.json}"
HISTORY_MODE="${HISTORY_MODE:-no_history}"
INCLUDE_SOURCES="${INCLUDE_SOURCES:-single_step}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
ONLY_SUCCESSFUL_ACTIONS="${ONLY_SUCCESSFUL_ACTIONS:-false}"

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

mkdir -p "$OUTPUT_DIR"

builder_args=(
  --output "$OUTPUT_FILE"
  --history_mode "$HISTORY_MODE"
  --include_sources "$INCLUDE_SOURCES"
  --single_step_file "$SINGLE_STEP_FILE"
  --multi_step_file "$MULTI_STEP_FILE"
  --image_root "$IMAGE_ROOT"
  --max_samples "$MAX_SAMPLES"
)

if [[ "$ONLY_SUCCESSFUL_ACTIONS" == "true" ]]; then
  builder_args+=(--only_successful_actions)
fi

cd "$REPO_ROOT"
python -m vagen.envs.navigation.create_datasets.build_ebnav_sft "${builder_args[@]}"
echo "data ready: $OUTPUT_FILE"
