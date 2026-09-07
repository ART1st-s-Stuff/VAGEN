#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-base
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=12:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-base-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-base-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
export SERVER_PORT=${SERVER_PORT:-5000}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export WANDB_MODE=${WANDB_MODE:-offline}
export SERVER_USE_STATE_REWARD=${SERVER_USE_STATE_REWARD:-False}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-navigation_base}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-0}
export VAGEN_RUN_LOG=${VAGEN_RUN_LOG:-$VAGEN_ARTIFACT_ROOT/logs/$EXPERIMENT_NAME.log}

# shellcheck disable=SC1091
source "$VAGEN_REPO/scripts/superpod/artifact_env.sh"
# shellcheck disable=SC1091
source "$VAGEN_REPO/scripts/superpod/load_modules.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"
# shellcheck disable=SC1091
source "$VAGEN_REPO/scripts/superpod/configure_vulkan_env.sh"

cd "$VAGEN_REPO"
# shellcheck disable=SC1091
source scripts/superpod/start_local_server.sh
bash scripts/examples/vagen_base/navigation/run.sh
bash scripts/superpod/write_run_summary.sh navigation_base "bash scripts/examples/vagen_base/navigation/run.sh"
