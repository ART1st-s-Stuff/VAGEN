#!/bin/bash
#SBATCH --job-name=vagen-vulkan-check
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/vulkan-check-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/vulkan-check-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/hligb/test_lu/VAGEN-navigation-repro}"

cd "$REPO_ROOT"
source scripts/superpod/artifact_env.sh
source scripts/superpod/load_modules.sh

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-vagen_nav}"
source scripts/superpod/configure_vulkan_env.sh

mkdir -p "$VAGEN_ARTIFACT_ROOT/logs"

echo "date: $(date -Is)"
echo "host: $(hostname)"
echo "cuda visible devices: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "conda prefix: ${CONDA_PREFIX:-unset}"
echo "ld library path: ${LD_LIBRARY_PATH:-unset}"
echo "vk icd filenames: ${VK_ICD_FILENAMES:-unset}"
echo "xdg runtime dir: ${XDG_RUNTIME_DIR:-unset}"
echo "ai2thor home: ${VAGEN_AI2THOR_HOME:-unset}"
echo

echo "== GPU =="
nvidia-smi || true
echo

echo "== Vulkan loader/libs =="
which vulkaninfo || true
ldconfig -p 2>/dev/null | grep -i vulkan || true
find "${CONDA_PREFIX:-/missing}" -maxdepth 3 \( -name 'libvulkan.so*' -o -name '*vulkan*icd*.json' \) -print 2>/dev/null || true
echo

echo "== Vulkan ICDs =="
ls -l /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>/dev/null || true
find /usr/share/vulkan /etc/vulkan -maxdepth 4 -type f -print 2>/dev/null || true
echo

echo "== vulkaninfo =="
vulkaninfo --summary || true
echo

echo "== conda packages =="
conda list | grep -Ei 'vulkan|mesa|libglvnd|glx|nvidia' || true
echo

echo "== AI2-THOR platform validation =="
if [ -n "${VAGEN_AI2THOR_HOME:-}" ]; then
  export HOME="$VAGEN_AI2THOR_HOME"
fi
python - <<'PY'
import ai2thor.platform
from ai2thor.controller import Controller
import os

print("effective HOME:", os.path.expanduser("~"))

try:
    controller = Controller(
        platform=ai2thor.platform.CloudRendering,
        width=300,
        height=300,
        gpu_device=0,
        server_timeout=120,
        server_start_timeout=120,
    )
    event = controller.step(action="Initialize")
    print("controller ok:", bool(event))
    controller.stop()
except Exception as exc:
    print("controller error:", repr(exc))
    raise
PY
