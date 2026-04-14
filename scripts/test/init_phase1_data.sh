#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/ELEN/datasets/navigation}"
SINGLE_STEP_FILE="${SINGLE_STEP_FILE:-$DATA_ROOT/eb-nav_dataset_single_step.json}"
MULTI_STEP_FILE="${MULTI_STEP_FILE:-$DATA_ROOT/eb-nav_dataset_multi_step.json}"
IMAGE_ROOT="${IMAGE_ROOT:-$DATA_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/phase1_debug}"
OUTPUT_FILE="${OUTPUT_FILE:-$OUTPUT_DIR/ebnav_small.json}"
HISTORY_MODE="${HISTORY_MODE:-no_history}"
INCLUDE_SOURCES="${INCLUDE_SOURCES:-single_step}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
ONLY_SUCCESSFUL_ACTIONS="${ONLY_SUCCESSFUL_ACTIONS:-false}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/verl:$PYTHONPATH"
else
  export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/verl"
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
"$PYTHON_BIN" -m vagen.envs.navigation.create_datasets.build_ebnav_sft "${builder_args[@]}"
echo "data ready: $OUTPUT_FILE"
