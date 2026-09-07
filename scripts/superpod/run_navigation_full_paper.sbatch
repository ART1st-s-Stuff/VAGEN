#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-full
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=36:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-full-retry2-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-full-retry2-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
export SERVER_PORT=${SERVER_PORT:-$((5000 + ${SLURM_JOB_ID:-0} % 1000))}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_full_paper_20260621}
export SERVER_USE_STATE_REWARD=${SERVER_USE_STATE_REWARD:-True}
export SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-8}
export SERVER_NAVIGATION_DEVICES=${SERVER_NAVIGATION_DEVICES:-0,1,2,3}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-navigation_full_paper_retry2_online_20260621}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-0}
export SAVE_FREQ=${SAVE_FREQ:-300}
export TEST_FREQ=${TEST_FREQ:-20}
export ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT:-900}
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
bash scripts/examples/vagen_full/navigation/run.sh
bash scripts/superpod/write_run_summary.sh "$EXPERIMENT_NAME" "WANDB_MODE=$WANDB_MODE EXPERIMENT_NAME=$EXPERIMENT_NAME bash scripts/examples/vagen_full/navigation/run.sh"
