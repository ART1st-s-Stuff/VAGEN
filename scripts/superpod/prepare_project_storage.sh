#!/usr/bin/env bash
# Route VAGEN artifacts and caches away from /home and into /project.

if [ -n "${VAGEN_PROJECT_STORAGE_PREPARED:-}" ] && [ -z "${SLURM_JOB_ID:-}" ]; then
  return 0 2>/dev/null || exit 0
fi
export VAGEN_PROJECT_STORAGE_PREPARED=1

VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
_VAGEN_JOB_ID=${SLURM_JOB_ID:-manual-$$}

export VAGEN_ARTIFACT_ROOT
export VAGEN_DATA_ROOT=${VAGEN_DATA_ROOT:-$VAGEN_ARTIFACT_ROOT/data}
export VAGEN_CHECKPOINT_ROOT=${VAGEN_CHECKPOINT_ROOT:-$VAGEN_ARTIFACT_ROOT/checkpoints}
export VAGEN_LOG_ROOT=${VAGEN_LOG_ROOT:-$VAGEN_ARTIFACT_ROOT/logs}
export HF_HOME=${HF_HOME:-$VAGEN_ARTIFACT_ROOT/hf}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/transformers}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$VAGEN_ARTIFACT_ROOT/xdg-cache}
export TORCH_HOME=${TORCH_HOME:-$VAGEN_ARTIFACT_ROOT/torch}
export WANDB_DIR=${WANDB_DIR:-$VAGEN_ARTIFACT_ROOT/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-$VAGEN_ARTIFACT_ROOT/wandb-cache}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-$VAGEN_ARTIFACT_ROOT/wandb-config}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$VAGEN_ARTIFACT_ROOT/pip-cache}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-$VAGEN_ARTIFACT_ROOT/conda/pkgs}
export VAGEN_FAST_TMP_ROOT=${VAGEN_FAST_TMP_ROOT:-/project/peilab/hligb/vn-tmp}
if [ -n "${SLURM_JOB_ID:-}" ]; then
  export RAY_TMPDIR="$VAGEN_FAST_TMP_ROOT/ray/$_VAGEN_JOB_ID"
  export TMPDIR="$VAGEN_FAST_TMP_ROOT/tmp/$_VAGEN_JOB_ID"
  export TRITON_CACHE_DIR="$VAGEN_FAST_TMP_ROOT/triton/$_VAGEN_JOB_ID"
  export TORCHINDUCTOR_CACHE_DIR="$VAGEN_FAST_TMP_ROOT/inductor/$_VAGEN_JOB_ID"
  export VAGEN_AI2THOR_HOME="$VAGEN_ARTIFACT_ROOT/ai2thor-home/$_VAGEN_JOB_ID"
else
  export RAY_TMPDIR=${RAY_TMPDIR:-$VAGEN_FAST_TMP_ROOT/ray/$_VAGEN_JOB_ID}
  export TMPDIR=${TMPDIR:-$VAGEN_FAST_TMP_ROOT/tmp/$_VAGEN_JOB_ID}
  export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$VAGEN_FAST_TMP_ROOT/triton/$_VAGEN_JOB_ID}
  export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$VAGEN_FAST_TMP_ROOT/inductor/$_VAGEN_JOB_ID}
  export VAGEN_AI2THOR_HOME=${VAGEN_AI2THOR_HOME:-$VAGEN_ARTIFACT_ROOT/ai2thor-home/$_VAGEN_JOB_ID}
fi
export VAGEN_AI2THOR_SHARED_HOME=${VAGEN_AI2THOR_SHARED_HOME:-$VAGEN_ARTIFACT_ROOT/ai2thor-home}

mkdir -p \
  "$VAGEN_DATA_ROOT" "$VAGEN_CHECKPOINT_ROOT" "$VAGEN_LOG_ROOT" \
  "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" \
  "$XDG_CACHE_HOME" "$TORCH_HOME" "$WANDB_DIR" "$WANDB_CACHE_DIR" \
  "$WANDB_CONFIG_DIR" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" \
  "$VAGEN_FAST_TMP_ROOT" "$RAY_TMPDIR" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" \
  "$VAGEN_AI2THOR_HOME" "$VAGEN_AI2THOR_SHARED_HOME" "$VAGEN_AI2THOR_SHARED_HOME/.ai2thor/releases" "$VAGEN_ARTIFACT_ROOT/wandb-home"

_prepare_link_dir() {
  local src="$1"
  local dst="$2"
  mkdir -p "$(dirname "$src")" "$dst"
  if [ -L "$src" ]; then
    local current
    current=$(readlink "$src" || true)
    [ "$current" = "$dst" ] && return 0
    rm -f "$src"
  fi
  if [ -e "$src" ]; then
    if [ -d "$src" ]; then
      shopt -s dotglob nullglob
      local item base
      for item in "$src"/*; do
        base=$(basename "$item")
        if [ ! -e "$dst/$base" ]; then
          mv "$item" "$dst/"
        else
          echo "prepare_project_storage: keep existing $dst/$base; leave $item in place" >&2
        fi
      done
      shopt -u dotglob nullglob
      rmdir "$src" 2>/dev/null || mv "$src" "$dst/migrated-$(date +%Y%m%d%H%M%S)-$(basename "$src")"
    else
      mv "$src" "$dst/$(basename "$src")"
    fi
  fi
  ln -s "$dst" "$src"
}

if command -v flock >/dev/null 2>&1; then
  mkdir -p "$VAGEN_ARTIFACT_ROOT"
  exec 91>"$VAGEN_ARTIFACT_ROOT/.prepare-storage.lock"
  flock -w 120 91 || echo "prepare_project_storage: could not acquire storage lock; continuing" >&2
fi

if [ "${VAGEN_PREPARE_HOME_LINKS:-1}" = "1" ] && [ -n "${HOME:-}" ]; then
  _prepare_link_dir "$HOME/.cache/huggingface" "$HF_HOME"
  _prepare_link_dir "$HOME/.cache/torch" "$TORCH_HOME"
  _prepare_link_dir "$HOME/wandb" "$VAGEN_ARTIFACT_ROOT/wandb-home"
  _prepare_link_dir "$HOME/.ai2thor" "$VAGEN_ARTIFACT_ROOT/ai2thor-home/default"
fi

if [ "${VAGEN_PREPARE_REPO_LINKS:-1}" = "1" ]; then
  _SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  _REPO_DIR=${VAGEN_REPO:-$(cd "$_SCRIPT_DIR/../.." && pwd)}
  if [ -d "$_REPO_DIR" ]; then
    _prepare_link_dir "$_REPO_DIR/data" "$VAGEN_DATA_ROOT"
    _prepare_link_dir "$_REPO_DIR/checkpoints" "$VAGEN_CHECKPOINT_ROOT"
    _prepare_link_dir "$_REPO_DIR/wandb" "$WANDB_DIR"
  fi
fi

df -h "$VAGEN_ARTIFACT_ROOT" "${HOME:-/home}" 2>/dev/null || true
echo "prepare_project_storage: artifacts=$VAGEN_ARTIFACT_ROOT job=$_VAGEN_JOB_ID"
echo "prepare_project_storage: checkpoints=$VAGEN_CHECKPOINT_ROOT data=$VAGEN_DATA_ROOT wandb=$WANDB_DIR"
echo "prepare_project_storage: ray_tmp=$RAY_TMPDIR tmp=$TMPDIR"
