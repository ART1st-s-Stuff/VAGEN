#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-full-preflight4-tg
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-full-preflight4-together-save25-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-full-preflight4-together-save25-%j.err

set -euo pipefail
export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
source "$VAGEN_REPO/scripts/superpod/prepare_project_storage.sh"
source "$VAGEN_REPO/scripts/superpod/load_modules.sh" >/dev/null 2>&1 || true
cd "$VAGEN_REPO"

srun --ntasks=1 --gres=gpu:4 bash scripts/superpod/preflight_navigation_full_4gpu_together_save25.sh
