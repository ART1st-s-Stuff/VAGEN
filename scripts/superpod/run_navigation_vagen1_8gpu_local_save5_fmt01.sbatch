#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-vagen1-8g-fmt01
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=768G
#SBATCH --time=12:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-8gpu-local-fmt01-save5-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-8gpu-local-fmt01-save5-%j.err

set -euo pipefail

export FORMAT_REWARD=${FORMAT_REWARD:-0.1}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_vagen1_local8gpu_fmt01_save5_20260725}
export WANDB_TAGS=${WANDB_TAGS:-vagen,navigation,vagen1,step1,local-env,8gpu,b16,rmb2,total60,val5,save5,raw-samples,answer-only,fmt01,no-dense}
export RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-navigation_vagen1_local8gpu_b16_rmb2_val5_save5_raw_fmt01_${RUN_STAMP}}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}

bash "${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505}/scripts/superpod/run_navigation_vagen1_8gpu_local_save5.sbatch"
