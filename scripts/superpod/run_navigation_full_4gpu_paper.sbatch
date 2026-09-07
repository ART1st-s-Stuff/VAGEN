#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-full4
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=36:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-full4-paper-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-full4-paper-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
source "$VAGEN_REPO/scripts/superpod/prepare_project_storage.sh"

export SERVER_PORT=${SERVER_PORT:-$((5000 + ${SLURM_JOB_ID:-0} % 1000))}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_full_paper_repro_20260702_ckpt50}
export WANDB_TAGS=${WANDB_TAGS:-vagen,navigation,full,4gpu,paper-repro,full-only,ckpt50}
export SERVER_USE_STATE_REWARD=${SERVER_USE_STATE_REWARD:-True}
export SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-8}
export SERVER_NAVIGATION_DEVICES=${SERVER_NAVIGATION_DEVICES:-0,1,2,3}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-navigation_full_4gpu_paper_repro_20260702_ckpt50}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-1}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-4}
export ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-64}
export ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT:-900}
export ROLLOUT_LIMIT_MM_PER_PROMPT=${ROLLOUT_LIMIT_MM_PER_PROMPT:-20}
export SAVE_FREQ=${SAVE_FREQ:-50}
export TEST_FREQ=${TEST_FREQ:-20}
export REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE:-False}
export VAGEN_RUN_LOG=${VAGEN_RUN_LOG:-$VAGEN_LOG_ROOT/$EXPERIMENT_NAME.log}

source "$VAGEN_REPO/scripts/superpod/load_modules.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"
source "$VAGEN_REPO/scripts/superpod/configure_vulkan_env.sh"

cd "$VAGEN_REPO"
source scripts/superpod/start_local_server.sh
bash scripts/examples/vagen_full/navigation/run.sh
bash scripts/superpod/write_run_summary.sh "$EXPERIMENT_NAME" \
  "WANDB_MODE=$WANDB_MODE EXPERIMENT_NAME=$EXPERIMENT_NAME N_GPUS_PER_NODE=$N_GPUS_PER_NODE SAVE_FREQ=$SAVE_FREQ REMOVE_PREVIOUS_CKPT_IN_SAVE=$REMOVE_PREVIOUS_CKPT_IN_SAVE bash scripts/examples/vagen_full/navigation/run.sh"
