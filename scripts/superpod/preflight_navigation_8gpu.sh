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
  echo "preflight: expected at least 8 GPUs, saw $GPU_COUNT" >&2
  nvidia-smi >&2 || true
  exit 1
fi

delete_preflight_wandb() {
  (cd /tmp && python "$VAGEN_REPO/scripts/superpod/cleanup_wandb_runs.py" \
    --project "${WANDB_PROJECT_PATH:-lukieluu6-city-university-of-hong-kong/vagen_navigation_repro}" \
    --prefix "zz_delete_preflight_" \
    --states finished failed crashed killed running preempted 2>/dev/null || true)
}

run_smoke() {
  local kind="$1"
  local state_reward="$2"
  local run_script="$3"
  local env_config="$4"
  local timestamp
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  export SERVER_PORT=$((5200 + RANDOM % 500))
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
  export SERVER_NAVIGATION_DEVICES=${SERVER_NAVIGATION_DEVICES:-0,1,2,3,4,5,6,7}
  export SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-8}
  export SERVER_USE_STATE_REWARD="$state_reward"
  export WANDB_MODE=${WANDB_MODE:-online}
  export WANDB_RUN_GROUP=navigation_preflight_20260621
  export EXPERIMENT_NAME="zz_delete_preflight_${kind}_8gpu_${timestamp}"
  export WANDB_NAME="$EXPERIMENT_NAME"
  export FORCE_GEN_DATA=1
  export ENV_CONFIG_PATH="$env_config"
  export TOTAL_TRAINING_STEPS=1
  export TRAIN_BATCH_SIZE=8
  export VAL_BATCH_SIZE=1
  export PPO_MINI_BATCH_SIZE=8
  export N_GPUS_PER_NODE=8
  export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=4
  export ROLLOUT_MINI_BATCH_SIZE=1
  export ROLLOUT_TIMEOUT=900
  export SAVE_FREQ=-1
  export TEST_FREQ=1
  export REMOVE_PREVIOUS_CKPT_IN_SAVE=True
  export VAGEN_RUN_LOG="$VAGEN_LOG_ROOT/$EXPERIMENT_NAME.log"
  export VAGEN_AI2THOR_HOME="$VAGEN_ARTIFACT_ROOT/ai2thor-home/preflight-$kind-$timestamp"
  mkdir -p "$VAGEN_AI2THOR_HOME"
  echo "preflight: starting $kind smoke on port $SERVER_PORT"
  source scripts/superpod/start_local_server.sh
  trap 'cleanup_vagen_server || true; delete_preflight_wandb' EXIT
  bash "$run_script"
  cleanup_vagen_server || true
  trap 'delete_preflight_wandb' EXIT
  echo "preflight: $kind smoke completed"
}

trap 'delete_preflight_wandb' EXIT
PREFLIGHT_TARGET=${PREFLIGHT_TARGET:-full}
case "$PREFLIGHT_TARGET" in
  all)
    run_smoke base False scripts/examples/vagen_base/navigation/run.sh scripts/examples/vagen_base/navigation/env_config_smoke.yaml
    run_smoke full True scripts/examples/vagen_full/navigation/run.sh scripts/examples/vagen_full/navigation/env_config_smoke.yaml
    echo "preflight: Base and Full 8GPU smoke completed"
    ;;
  base)
    run_smoke base False scripts/examples/vagen_base/navigation/run.sh scripts/examples/vagen_base/navigation/env_config_smoke.yaml
    echo "preflight: Base 8GPU smoke completed"
    ;;
  full)
    run_smoke full True scripts/examples/vagen_full/navigation/run.sh scripts/examples/vagen_full/navigation/env_config_smoke.yaml
    echo "preflight: Full 8GPU smoke completed"
    ;;
  *)
    echo "preflight: unknown PREFLIGHT_TARGET=$PREFLIGHT_TARGET" >&2
    exit 2
    ;;
esac
delete_preflight_wandb
