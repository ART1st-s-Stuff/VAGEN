#!/usr/bin/env bash
set -euo pipefail

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_vagen1_engine_smoke_limit8_batch8_smokedata_20260722}
export WANDB_TAGS=${WANDB_TAGS:-vagen,navigation,vagen1,step1,engine-smoke,normal,4gpu,limit8,batch8,smokedata}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-1}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export ENV_CONFIG_PATH=${ENV_CONFIG_PATH:-$VAGEN_REPO/scripts/examples/vagen_base/navigation_vagen1/env_config_smoke.yaml}
export SAVE_FREQ=${SAVE_FREQ:--1}
export TEST_FREQ=${TEST_FREQ:-1}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-1}
export REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE:-True}
export RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}

variants=(
  vagenrt_gpu04_limit8
  vagenrt_gpu06_limit8
  eager_gpu04_limit8
  eager_gpu06_limit8
  eager_free_gpu04_limit8
  eager_chunk_gpu04_limit8
  failed_minus_limit8_gpu06
  tp1_diag_gpu06_limit8
)

for variant in "${variants[@]}"; do
  job=$(sbatch --parsable --time=02:00:00 \
    --export=ALL,VAGEN1_VARIANT="$variant",RUN_STAMP="$RUN_STAMP" \
    scripts/superpod/run_navigation_vagen1_4gpu.sbatch)
  echo "$variant job=$job"
done
