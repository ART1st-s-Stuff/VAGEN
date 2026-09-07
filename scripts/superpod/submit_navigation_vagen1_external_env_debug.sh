#!/usr/bin/env bash
set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-$(pwd)}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
source "$VAGEN_REPO/scripts/superpod/prepare_project_storage.sh"
source "$VAGEN_REPO/scripts/superpod/load_modules.sh"

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_vagen1_external_env_debug5_20260722}
export WANDB_TAGS=${WANDB_TAGS:-vagen,navigation,vagen1,step1,external-ai2thor,debug5,normal,4gpu}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-6}
export SAVE_FREQ=${SAVE_FREQ:--1}
export TEST_FREQ=${TEST_FREQ:-5}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-False}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-1}
export REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE:-True}
export ENV_CONFIG_PATH=${ENV_CONFIG_PATH:-$VAGEN_REPO/scripts/examples/vagen_base/navigation_vagen1/env_config_speed.yaml}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
export SERVER_WAIT_SECONDS=${SERVER_WAIT_SECONDS:-7200}
export ENV_SERVER_TIME=${ENV_SERVER_TIME:-04:00:00}
export TRAIN_TIME=${TRAIN_TIME:-02:00:00}
export SERVER_PREWARM_AI2THOR=${SERVER_PREWARM_AI2THOR:-0}
export RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}

variants=("$@")
if [ "${#variants[@]}" -eq 0 ]; then
  variants=(
    external_b16_rmb16_w4_turn20
    external_b32_rmb16_w4_turn20
  )
fi

submit_log="$VAGEN_LOG_ROOT/vagen1-external-env-debug-submit-${RUN_STAMP}.log"

idx=0
for variant in "${variants[@]}"; do
  idx=$((idx + 1))
  (
    export VAGEN1_VARIANT="$variant"
    source "$VAGEN_REPO/scripts/superpod/configure_navigation_vagen1_variant.sh"

    server_session="navigation_vagen1_${variant}_${RUN_STAMP}_${idx}"
    ready_file="$VAGEN_LOG_ROOT/navigation-ai2thor-server-${server_session}.env"

    server_job=$(sbatch --parsable --time="$ENV_SERVER_TIME" \
      --export=ALL,VAGEN_REPO="$VAGEN_REPO",VAGEN_ARTIFACT_ROOT="$VAGEN_ARTIFACT_ROOT",SERVER_SESSION_ID="$server_session",SERVER_READY_FILE="$ready_file",SERVER_USE_STATE_REWARD=False,SERVER_PREWARM_AI2THOR="$SERVER_PREWARM_AI2THOR",SERVER_NAVIGATION_MAX_WORKERS="$SERVER_NAVIGATION_MAX_WORKERS" \
      scripts/superpod/run_navigation_vagen1_ai2thor_server.sbatch)
    if [[ ! "$server_job" =~ ^[0-9]+$ ]]; then
      echo "sbatch did not return a numeric env server job id: $server_job" >&2
      exit 4
    fi

    train_job=$(sbatch --parsable --time="$TRAIN_TIME" --dependency=after:"$server_job" \
      --export=ALL,VAGEN_REPO="$VAGEN_REPO",VAGEN_ARTIFACT_ROOT="$VAGEN_ARTIFACT_ROOT",SERVER_SESSION_ID="$server_session",SERVER_READY_FILE="$ready_file",SERVER_JOB_ID="$server_job",VAGEN1_VARIANT="$variant",RUN_STAMP="$RUN_STAMP",TOTAL_TRAINING_STEPS="$TOTAL_TRAINING_STEPS",SAVE_FREQ="$SAVE_FREQ",TEST_FREQ="$TEST_FREQ",VAL_BEFORE_TRAIN="$VAL_BEFORE_TRAIN",FINAL_VAL_AFTER_TRAIN="$FINAL_VAL_AFTER_TRAIN",FORCE_GEN_DATA="$FORCE_GEN_DATA",REMOVE_PREVIOUS_CKPT_IN_SAVE="$REMOVE_PREVIOUS_CKPT_IN_SAVE",ENV_CONFIG_PATH="$ENV_CONFIG_PATH",VAL_BATCH_SIZE="$VAL_BATCH_SIZE",SERVER_WAIT_SECONDS="$SERVER_WAIT_SECONDS" \
      scripts/superpod/run_navigation_vagen1_4gpu_external_server.sbatch)
    if [[ ! "$train_job" =~ ^[0-9]+$ ]]; then
      echo "sbatch did not return a numeric training job id: $train_job" >&2
      exit 5
    fi

    echo "$variant env_job=$server_job train_job=$train_job ready_file=$ready_file train_batch=$TRAIN_BATCH_SIZE rollout_mini=$ROLLOUT_MINI_BATCH_SIZE server_workers=$SERVER_NAVIGATION_MAX_WORKERS"
  )
done | tee "$submit_log"

echo "submit_log=$submit_log"
