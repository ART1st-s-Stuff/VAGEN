#!/usr/bin/env bash

if [ -n "${CONDA_PREFIX:-}" ] && [ -d "$CONDA_PREFIX/lib" ]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

if [ -z "${VK_ICD_FILENAMES:-}" ] && [ -r /usr/share/vulkan/icd.d/nvidia_icd.json ]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

if [ -z "${XDG_RUNTIME_DIR:-}" ]; then
  export XDG_RUNTIME_DIR="/tmp/vagen-runtime-${USER:-user}-${SLURM_JOB_ID:-manual}"
  mkdir -p "$XDG_RUNTIME_DIR"
  chmod 700 "$XDG_RUNTIME_DIR" 2>/dev/null || true
fi

if [ -n "${VAGEN_ARTIFACT_ROOT:-}" ]; then
  export VAGEN_AI2THOR_HOME="${VAGEN_AI2THOR_HOME:-$VAGEN_ARTIFACT_ROOT/ai2thor-home}"
  mkdir -p "$VAGEN_AI2THOR_HOME"
fi
