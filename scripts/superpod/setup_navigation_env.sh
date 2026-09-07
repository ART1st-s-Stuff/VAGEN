#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
VERL_ROOT=${VAGEN_VERL_ROOT:-${ARTIFACT_ROOT}/verl}
CONDA_ENV=${VAGEN_CONDA_ENV:-vagen_nav}
VERL_REF=${VAGEN_VERL_REF:-vagen-legacy}
RECREATE_ENV=${VAGEN_RECREATE_ENV:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/artifact_env.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/load_modules.sh"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Load conda/miniconda on SuperPOD, then rerun." >&2
  exit 1
fi

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

if [ "$RECREATE_ENV" = "1" ] && conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda env remove -n "$CONDA_ENV" -y || true
fi

CONDA_ENV_DIR="$CONDA_BASE/envs/$CONDA_ENV"
if [ "$RECREATE_ENV" = "1" ] && [ -d "$CONDA_ENV_DIR" ]; then
  case "$CONDA_ENV" in
    ""|*/*|*\\*) echo "refusing to remove unsafe conda env name: $CONDA_ENV" >&2; exit 1 ;;
  esac
  case "$CONDA_ENV_DIR" in
    "$CONDA_BASE"/envs/*) rm -rf "$CONDA_ENV_DIR" ;;
    *) echo "refusing to remove unsafe conda env path: $CONDA_ENV_DIR" >&2; exit 1 ;;
  esac
fi

if ! conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  conda create -n "$CONDA_ENV" python=3.10 -y
fi

conda activate "$CONDA_ENV"
conda install -y -c conda-forge vulkan-loader vulkan-tools
# shellcheck disable=SC1091
source "$SCRIPT_DIR/configure_vulkan_env.sh"
python -m pip install --upgrade pip setuptools==80.9.0 wheel

if [ ! -d "$VERL_ROOT/.git" ]; then
  git clone https://github.com/JamesKrW/verl.git "$VERL_ROOT"
fi

git -C "$VERL_ROOT" fetch --all --prune
git -C "$VERL_ROOT" checkout -B "$VERL_REF" "origin/$VERL_REF"
python -m pip install -e "$VERL_ROOT"

cd "$REPO_DIR"
bash scripts/install.sh

python -m pip install ai2thor==5.0.0 numpy==1.25.1

if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
  sudo -n apt-get update
  sudo -n apt-get install -y libvulkan1 vulkan-tools
else
  echo "passwordless sudo not available; using conda vulkan-loader/vulkan-tools and compute-node NVIDIA ICD." >&2
fi

python - <<'PY'
import ai2thor
import numpy
import vagen
print("vagen navigation environment import check ok")
print("ai2thor", getattr(ai2thor, "__version__", "unknown"))
print("numpy", numpy.__version__)
PY
