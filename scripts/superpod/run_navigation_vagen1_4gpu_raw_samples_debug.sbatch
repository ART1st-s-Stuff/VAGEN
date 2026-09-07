#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-vagen1-raw
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=02:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-raw-4gpu-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-raw-4gpu-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505}
cd "$VAGEN_REPO"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-navigation_vagen1_raw_samples_4gpu_debug_${timestamp}}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_vagen1_raw_samples_4gpu_debug_20260724}
export WANDB_TAGS=${WANDB_TAGS:-vagen,navigation,vagen1,single-action,raw-samples,debug,answer-only,fmt005}

export VAGEN1_VARIANT=${VAGEN1_VARIANT:-fmt005_answeronly}
export MAX_TURNS=${MAX_TURNS:-20}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-2}
export SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-1}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-4}
export TEST_FREQ=${TEST_FREQ:-2}
export SAVE_FREQ=${SAVE_FREQ:--1}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-False}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-1}
export ENV_CONFIG_PATH=${ENV_CONFIG_PATH:-$VAGEN_REPO/scripts/examples/vagen_base/navigation_vagen1/env_config_speed.yaml}
export RAW_SAMPLES_TO_LOG=${RAW_SAMPLES_TO_LOG:-8}
export RAW_SAMPLES_MAX_CHARS=${RAW_SAMPLES_MAX_CHARS:-2000}

bash "$VAGEN_REPO/scripts/superpod/run_navigation_vagen1_4gpu.sbatch"
