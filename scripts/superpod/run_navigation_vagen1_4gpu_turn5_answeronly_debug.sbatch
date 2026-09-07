#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-v1-t5-ao
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-4gpu-turn5-answeronly-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-4gpu-turn5-answeronly-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}

export VAGEN1_VARIANT=${VAGEN1_VARIANT:-turn5_answeronly_fmt005_rmb2_w1}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_vagen1_turn5_answeronly_4gpu_debug5_20260723}
export WANDB_TAGS=${WANDB_TAGS:-vagen,navigation,vagen1,step1,4gpu,turn5,answer-only,fmt005,rmb2,w1,debug5}
export RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-navigation_vagen1_turn5_answeronly_fmt005_rmb2_w1_4gpu_debug5_${RUN_STAMP}}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-6}
export TEST_FREQ=${TEST_FREQ:-5}
export SAVE_FREQ=${SAVE_FREQ:--1}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-False}
export REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE:-False}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-1}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
export ENV_CONFIG_PATH=${ENV_CONFIG_PATH:-$VAGEN_REPO/scripts/examples/vagen_base/navigation_vagen1/env_config_speed.yaml}
export VAGEN_RUN_LOG=${VAGEN_RUN_LOG:-$VAGEN_ARTIFACT_ROOT/logs/$EXPERIMENT_NAME.log}

bash "$VAGEN_REPO/scripts/superpod/run_navigation_vagen1_4gpu.sbatch"
