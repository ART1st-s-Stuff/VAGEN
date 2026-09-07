#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-full8-tg25
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=768G
#SBATCH --time=36:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-full8-together-save25-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-full8-together-save25-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
source "$VAGEN_REPO/scripts/superpod/prepare_project_storage.sh"

# Keep Unity/AI2-THOR per-job runtime files off /home. Shared releases/cache stay under /project.
export VAGEN_NODE_LOCAL_ROOT=${VAGEN_NODE_LOCAL_ROOT:-/tmp/${USER:-hligb}/vagen-navigation/${SLURM_JOB_ID:-manual}}
export VAGEN_AI2THOR_HOME=${VAGEN_LOCAL_AI2THOR_HOME:-$VAGEN_NODE_LOCAL_ROOT/ai2thor-home}
export RAY_TMPDIR=$VAGEN_NODE_LOCAL_ROOT/ray
export TMPDIR=$VAGEN_NODE_LOCAL_ROOT/tmp
export TRITON_CACHE_DIR=$VAGEN_NODE_LOCAL_ROOT/triton-cache
export TORCHINDUCTOR_CACHE_DIR=$VAGEN_NODE_LOCAL_ROOT/torchinductor-cache
mkdir -p "$VAGEN_AI2THOR_HOME" "$RAY_TMPDIR" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR"

export VAGEN_JUDGE_PROVIDER=${VAGEN_JUDGE_PROVIDER:-together}
export VAGEN_JUDGE_MODEL=${VAGEN_JUDGE_MODEL:-openai/gpt-oss-20b}
export VAGEN_JUDGE_REASONING_EFFORT=${VAGEN_JUDGE_REASONING_EFFORT:-low}
export VAGEN_JUDGE_MAX_TOKENS=${VAGEN_JUDGE_MAX_TOKENS:-512}
export SERVER_PORT=${SERVER_PORT:-$((5000 + ${SLURM_JOB_ID:-0} % 1000))}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_full_together_save25_20260703}
export WANDB_TAGS=${WANDB_TAGS:-vagen,navigation,full,8gpu,paper-repro,together,gpt-oss-20b,save25}
export SERVER_USE_STATE_REWARD=True
export SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-2}
export SERVER_NAVIGATION_DEVICES=${SERVER_NAVIGATION_DEVICES:-0,1,2,3,4,5,6,7}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-navigation_full_8gpu_paper_repro_20260703_together_gptoss20b_save25}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-0}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
export N_GPUS_PER_NODE=8
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-4}
export ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-16}
export ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT:-3600}
export SAVE_FREQ=${SAVE_FREQ:-25}
export TEST_FREQ=${TEST_FREQ:-25}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
export VAL_GENERATIONS_TO_LOG_TO_WANDB=${VAL_GENERATIONS_TO_LOG_TO_WANDB:-4}
export REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE:-True}
export VAGEN_RUN_LOG=${VAGEN_RUN_LOG:-$VAGEN_LOG_ROOT/$EXPERIMENT_NAME.log}

source "$VAGEN_REPO/scripts/superpod/load_modules.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"
source "$VAGEN_REPO/scripts/superpod/configure_vulkan_env.sh"

cd "$VAGEN_REPO"
source scripts/superpod/start_local_server.sh
bash scripts/examples/vagen_full/navigation/run.sh
bash scripts/superpod/write_run_summary.sh "$EXPERIMENT_NAME"   "WANDB_MODE=$WANDB_MODE EXPERIMENT_NAME=$EXPERIMENT_NAME N_GPUS_PER_NODE=$N_GPUS_PER_NODE SAVE_FREQ=$SAVE_FREQ TEST_FREQ=$TEST_FREQ VAGEN_JUDGE_PROVIDER=$VAGEN_JUDGE_PROVIDER VAGEN_JUDGE_MODEL=$VAGEN_JUDGE_MODEL REMOVE_PREVIOUS_CKPT_IN_SAVE=$REMOVE_PREVIOUS_CKPT_IN_SAVE bash scripts/examples/vagen_full/navigation/run.sh"
