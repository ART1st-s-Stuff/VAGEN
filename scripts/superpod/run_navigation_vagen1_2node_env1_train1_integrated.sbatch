#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-vagen1-2n-int
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-2node-env1-train1-integrated-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-2node-env1-train1-integrated-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}

export ENV_NNODES=${ENV_NNODES:-1}
export TRAIN_NNODES=${TRAIN_NNODES:-1}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export ENV_GPUS_PER_NODE=${ENV_GPUS_PER_NODE:-4}
export EXPECTED_RAY_GPUS=${EXPECTED_RAY_GPUS:-4}

export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_vagen1_integrated_2node_env1x4_train1x4_raw_val5_20260724}
export WANDB_TAGS=${WANDB_TAGS:-vagen,navigation,vagen1,step1,integrated,single-allocation,env1x4,train1x4,b16,rmb2,total60,val5,raw-samples,answer-only,fmt005}
export RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-navigation_vagen1_integrated2n_env1x4_train1x4_b16_rmb2_val5_raw_${RUN_STAMP}}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}

export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-60}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-2}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:-5}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-False}
export RAW_SAMPLES_TO_LOG=${RAW_SAMPLES_TO_LOG:-8}
export RAW_SAMPLES_MAX_CHARS=${RAW_SAMPLES_MAX_CHARS:-2000}
export REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE:-False}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-1}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}

export SERVER_WAIT_SECONDS=${SERVER_WAIT_SECONDS:-900}
export RAY_CLUSTER_WAIT_SECONDS=${RAY_CLUSTER_WAIT_SECONDS:-300}
export SERVER_USE_STATE_REWARD=${SERVER_USE_STATE_REWARD:-False}
export SERVER_PREWARM_AI2THOR=${SERVER_PREWARM_AI2THOR:-0}
export SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-4}
export SERVER_NAVIGATION_DEVICES=${SERVER_NAVIGATION_DEVICES:-0,1,2,3}

export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
export ROLLOUT_LIMIT_MM_PER_PROMPT=${ROLLOUT_LIMIT_MM_PER_PROMPT:-8}
export ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}
export ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
export ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}
export ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
export MAX_TURNS=${MAX_TURNS:-20}
export ROLLOUT_WINDOW_SIZE=${ROLLOUT_WINDOW_SIZE:-5}
export UPDATE_WINDOW_SIZE=${UPDATE_WINDOW_SIZE:-5}
export MAX_TRAJECTORY_LENGTH=${MAX_TRAJECTORY_LENGTH:-16000}
export ROLLOUT_MAX_TRAJECTORY_LENGTH=${ROLLOUT_MAX_TRAJECTORY_LENGTH:-5000}
export FORMAT_REWARD=${FORMAT_REWARD:-0.05}
export LOSS_MASK_MODE=${LOSS_MASK_MODE:-answer_only}
export ENV_CONFIG_PATH=${ENV_CONFIG_PATH:-$VAGEN_REPO/scripts/examples/vagen_base/navigation_vagen1/env_config_speed.yaml}
export VAGEN_RUN_LOG=${VAGEN_RUN_LOG:-$VAGEN_ARTIFACT_ROOT/logs/$EXPERIMENT_NAME.log}

bash "$VAGEN_REPO/scripts/superpod/run_navigation_vagen1_5node_integrated.sbatch"
