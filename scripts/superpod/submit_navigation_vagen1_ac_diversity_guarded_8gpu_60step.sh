#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/load_modules.sh"

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
export RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_vagen1_ac_guarded_60step_20260804}
export SLURM_EXCLUDE_NODES=${SLURM_EXCLUDE_NODES:-dgx-26,dgx-32,dgx-35,dgx-37}

experiment_name="navigation_vagen1_ac_diversity_guarded_turn20_ctx8192_60step_${RUN_STAMP}"
env_config="scripts/examples/vagen_base/navigation_vagen1/env_config_dense_ac_diversity_guarded.yaml"
tags="vagen,navigation,vagen1,step1,local-env,8gpu,turn20,ctx8192,w1,rmb2,default-loss,fmt005,ac-diversity,guarded,save15,raw-samples"

job_id=$(sbatch --parsable \
  --time=12:00:00 \
  --exclude="$SLURM_EXCLUDE_NODES" \
  --export=ALL,VAGEN_REPO="$VAGEN_REPO",VAGEN_ARTIFACT_ROOT="$VAGEN_ARTIFACT_ROOT",VAGEN1_VARIANT="eager_guard",RUN_STAMP="$RUN_STAMP",EXPERIMENT_NAME="$experiment_name",WANDB_NAME="$experiment_name",WANDB_RUN_GROUP="$WANDB_RUN_GROUP",WANDB_TAGS="$tags",ENV_CONFIG_PATH="$VAGEN_REPO/$env_config",TOTAL_TRAINING_STEPS=60,TEST_FREQ=15,SAVE_FREQ=15,VAL_BEFORE_TRAIN=False,FINAL_VAL_AFTER_TRAIN=False,FORCE_GEN_DATA=1,REMOVE_PREVIOUS_CKPT_IN_SAVE=True,RAW_SAMPLES_TO_LOG=8,RAW_SAMPLES_MAX_CHARS=2000,SERVER_NAVIGATION_MAX_WORKERS=1,ROLLOUT_MINI_BATCH_SIZE=2,MAX_TURNS=20,ROLLOUT_MAX_TRAJECTORY_LENGTH=8192,MAX_TRAJECTORY_LENGTH=16000,LOSS_MASK_MODE=default,FORMAT_REWARD=0.05,ROLLOUT_ENFORCE_EAGER=True \
  "$VAGEN_REPO/scripts/superpod/run_navigation_vagen1_8gpu_local_save5_dense_light.sbatch")

echo "| variant | job id | W&B name | env config |"
echo "| --- | --- | --- | --- |"
echo "| ac_diversity_guarded | $job_id | $experiment_name | $env_config |"
