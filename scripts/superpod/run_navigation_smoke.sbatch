#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-smoke
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-smoke-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-smoke-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
export SERVER_PORT=${SERVER_PORT:-5000}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export TOTAL_TRAINING_STEPS=1
export TRAIN_BATCH_SIZE=8
export VAL_BATCH_SIZE=2
export N_GPUS_PER_NODE=4
export PPO_MINI_BATCH_SIZE=8
export WANDB_MODE=${WANDB_MODE:-offline}
export SERVER_USE_STATE_REWARD=${SERVER_USE_STATE_REWARD:-True}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-smoke-vagen_full-navigation}
export ENV_CONFIG_PATH=${ENV_CONFIG_PATH:-$VAGEN_REPO/scripts/examples/vagen_full/navigation/env_config_smoke.yaml}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-1}
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
bash scripts/superpod/write_run_summary.sh navigation_smoke "TOTAL_TRAINING_STEPS=1 bash scripts/examples/vagen_full/navigation/run.sh"
