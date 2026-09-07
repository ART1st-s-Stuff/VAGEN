#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-b8-r2
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=768G
#SBATCH --time=36:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/nav-base8-retry2-online-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/nav-base8-retry2-online-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
export SERVER_PORT=${SERVER_PORT:-$((5000 + ${SLURM_JOB_ID:-0} % 1000))}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_base_reruns_20260621}
export SERVER_USE_STATE_REWARD=${SERVER_USE_STATE_REWARD:-False}
export SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-8}
export SERVER_NAVIGATION_DEVICES=${SERVER_NAVIGATION_DEVICES:-0,1,2,3,4,5,6,7}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-nav_base8_scale_retry2_online_20260621}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-0}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-4}
export ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-64}
export ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT:-900}
export SAVE_FREQ=${SAVE_FREQ:-300}
export TEST_FREQ=${TEST_FREQ:-20}
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
bash scripts/superpod/write_run_summary.sh "$EXPERIMENT_NAME" \
  "WANDB_MODE=$WANDB_MODE EXPERIMENT_NAME=$EXPERIMENT_NAME N_GPUS_PER_NODE=$N_GPUS_PER_NODE SAVE_FREQ=$SAVE_FREQ bash scripts/examples/vagen_base/navigation/run.sh"
