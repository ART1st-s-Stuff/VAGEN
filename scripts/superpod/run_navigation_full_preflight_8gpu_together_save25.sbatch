#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-full-preflight-tg
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=768G
#SBATCH --time=04:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-full-preflight-together-save25-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-full-preflight-together-save25-%j.err

set -euo pipefail
export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
source "$VAGEN_REPO/scripts/superpod/prepare_project_storage.sh"
source "$VAGEN_REPO/scripts/superpod/load_modules.sh" >/dev/null 2>&1 || true
cd "$VAGEN_REPO"

# Run the smoke test through srun inside the allocated node, matching the requested pre-submit gate.
srun --ntasks=1 --gres=gpu:8 bash scripts/superpod/preflight_navigation_full_together_save25.sh
