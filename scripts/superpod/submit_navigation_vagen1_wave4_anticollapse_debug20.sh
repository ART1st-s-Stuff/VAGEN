#!/usr/bin/env bash
set -euo pipefail

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_vagen1_anticollapse_debug20_20260722}
export WANDB_TAGS=${WANDB_TAGS:-vagen,navigation,vagen1,step1,anti-collapse,debug20,normal,4gpu}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-20}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:-20}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-1}
export REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE:-False}
export RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}

variants=(
  fmt01_defaultloss
  fmt01_answeronly
  fmt005_answeronly
  fmt01_answeronly_strict
)

for variant in "${variants[@]}"; do
  job=$(sbatch --parsable --time=08:00:00 \
    --export=ALL,VAGEN1_VARIANT="$variant",RUN_STAMP="$RUN_STAMP" \
    scripts/superpod/run_navigation_vagen1_4gpu.sbatch)
  echo "$variant job=$job"
done
