#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-server
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-server-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-server-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
export SERVER_PORT=${SERVER_PORT:-5000}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

# shellcheck disable=SC1091
source "$VAGEN_REPO/scripts/superpod/artifact_env.sh"
# shellcheck disable=SC1091
source "$VAGEN_REPO/scripts/superpod/load_modules.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"
# shellcheck disable=SC1091
source "$VAGEN_REPO/scripts/superpod/configure_vulkan_env.sh"

cd "$VAGEN_REPO"
if [ -n "${VAGEN_AI2THOR_HOME:-}" ]; then
  export HOME="$VAGEN_AI2THOR_HOME"
  rm -f "$HOME/.ai2thor/cuda-vulkan-mapping.json" \
    "$HOME/.ai2thor/cuda-vulkan-mapping.json.lock"
fi
python -m vagen.server.server server.port="$SERVER_PORT" use_state_reward=True
