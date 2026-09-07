#!/usr/bin/env bash
set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
source "$VAGEN_REPO/scripts/superpod/prepare_project_storage.sh"
source "$VAGEN_REPO/scripts/superpod/load_modules.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"
source "$VAGEN_REPO/scripts/superpod/configure_vulkan_env.sh"
cd "$VAGEN_REPO"
mkdir -p "$VAGEN_LOG_ROOT"

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')
if [ "$GPU_COUNT" -lt 8 ]; then
  echo "preflight together full: expected at least 8 GPUs, saw $GPU_COUNT" >&2
  nvidia-smi >&2 || true
  exit 1
fi

export VAGEN_JUDGE_PROVIDER=${VAGEN_JUDGE_PROVIDER:-together}
export VAGEN_JUDGE_MODEL=${VAGEN_JUDGE_MODEL:-openai/gpt-oss-20b}
export VAGEN_JUDGE_REASONING_EFFORT=${VAGEN_JUDGE_REASONING_EFFORT:-low}
export VAGEN_JUDGE_MAX_TOKENS=${VAGEN_JUDGE_MAX_TOKENS:-512}
export SERVER_PORT=${SERVER_PORT:-$((5400 + RANDOM % 400))}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export SERVER_NAVIGATION_DEVICES=${SERVER_NAVIGATION_DEVICES:-0,1,2,3,4,5,6,7}
export SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-2}
export SERVER_USE_STATE_REWARD=True
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_RUN_GROUP=navigation_full_together_preflight_20260703
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-zz_delete_preflight_full_8gpu_together_gptoss20b_$(date -u +%Y%m%dT%H%M%SZ)}
export WANDB_NAME="$EXPERIMENT_NAME"
export FORCE_GEN_DATA=1
export ENV_CONFIG_PATH=scripts/examples/vagen_full/navigation/env_config_smoke.yaml
export TOTAL_TRAINING_STEPS=1
export TRAIN_BATCH_SIZE=8
export VAL_BATCH_SIZE=1
export PPO_MINI_BATCH_SIZE=8
export N_GPUS_PER_NODE=8
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=4
export ROLLOUT_MINI_BATCH_SIZE=1
export ROLLOUT_TIMEOUT=1800
export SAVE_FREQ=-1
export TEST_FREQ=1
export VAL_BEFORE_TRAIN=False
export VAL_GENERATIONS_TO_LOG_TO_WANDB=1
export REMOVE_PREVIOUS_CKPT_IN_SAVE=True
export VAGEN_AI2THOR_HOME=${VAGEN_AI2THOR_HOME:-/tmp/${USER:-hligb}/vagen-navigation/${SLURM_JOB_ID:-manual}/ai2thor-preflight}
export VAGEN_RUN_LOG="$VAGEN_LOG_ROOT/$EXPERIMENT_NAME.log"
mkdir -p "$VAGEN_AI2THOR_HOME"

delete_preflight_wandb() {
  (cd /tmp && python "$VAGEN_REPO/scripts/superpod/cleanup_wandb_runs.py" \
    --project "${WANDB_PROJECT_PATH:-lukieluu6-city-university-of-hong-kong/vagen_navigation_repro}" \
    --prefix "zz_delete_preflight_" \
    --states finished failed crashed killed running preempted 2>/dev/null || true)
}

trap 'cleanup_vagen_server || true; delete_preflight_wandb' EXIT

echo "preflight together full: provider=$VAGEN_JUDGE_PROVIDER model=$VAGEN_JUDGE_MODEL port=$SERVER_PORT"
source scripts/superpod/start_local_server.sh
bash scripts/examples/vagen_full/navigation/run.sh
cleanup_vagen_server || true
delete_preflight_wandb
echo "preflight together full: completed"
