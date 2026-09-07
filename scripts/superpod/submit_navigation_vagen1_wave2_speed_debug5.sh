#!/usr/bin/env bash
set -euo pipefail

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_vagen1_speed_debug5_vagenrt06_limit8_total6_20260722}
export WANDB_TAGS=${WANDB_TAGS:-vagen,navigation,vagen1,step1,speed-debug5,actual5,normal,4gpu,vagenrt06,limit8}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-6}
export SAVE_FREQ=${SAVE_FREQ:--1}
export TEST_FREQ=${TEST_FREQ:-5}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-False}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-1}
export REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE:-True}
export ENV_CONFIG_PATH=${ENV_CONFIG_PATH:-$VAGEN_REPO/scripts/examples/vagen_base/navigation_vagen1/env_config_speed.yaml}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
export RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}

variants=(
  speed_b8_rmb4_w1_turn20
  speed_b8_rmb8_w1_turn20
  speed_b16_rmb8_w2_turn20
  speed_b16_rmb16_w4_turn20
  speed_b32_rmb8_w4_turn20
  speed_b32_rmb16_w4_turn20
  speed_b64_rmb16_w4_turn20
  speed_b64_rmb32_w4_turn20
)

for variant in "${variants[@]}"; do
  job=$(sbatch --parsable --time=04:00:00 \
    --export=ALL,VAGEN1_VARIANT="$variant",RUN_STAMP="$RUN_STAMP" \
    scripts/superpod/run_navigation_vagen1_4gpu.sbatch)
  echo "$variant job=$job"
done
