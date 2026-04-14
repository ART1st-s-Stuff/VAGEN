#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-/project}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

MODEL_REPO_ID="${MODEL_REPO_ID:-Qwen/Qwen2.5-VL-3B-Instruct}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/models/Qwen2.5-VL-3B-Instruct}"
export MODEL_REPO_ID
export MODEL_DIR

export HF_HOME="${HF_HOME:-$PROJECT_ROOT/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

mkdir -p "$MODEL_DIR"
mkdir -p "$HF_HOME"

"$PYTHON_BIN" - <<'PY'
import os
from huggingface_hub import snapshot_download

repo_id = os.environ["MODEL_REPO_ID"]
local_dir = os.environ["MODEL_DIR"]

snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"model ready: {local_dir}")
PY
