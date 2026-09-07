#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-ai2thor
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-ai2thor-server-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-ai2thor-server-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
export SERVER_SESSION_ID=${SERVER_SESSION_ID:-navigation_vagen1_ai2thor_${SLURM_JOB_ID:-manual}}
export SERVER_PORT=${SERVER_PORT:-$((7000 + ${SLURM_JOB_ID:-0} % 1000))}
export SERVER_USE_STATE_REWARD=${SERVER_USE_STATE_REWARD:-False}
export SERVER_PREWARM_AI2THOR=${SERVER_PREWARM_AI2THOR:-0}
export SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-4}
export SERVER_NAVIGATION_DEVICES=${SERVER_NAVIGATION_DEVICES:-0,1,2,3}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

source "$VAGEN_REPO/scripts/superpod/prepare_project_storage.sh"
source "$VAGEN_REPO/scripts/superpod/load_modules.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"

export SERVER_READY_FILE=${SERVER_READY_FILE:-$VAGEN_LOG_ROOT/navigation-ai2thor-server-${SERVER_SESSION_ID}.env}
export SERVER_LOG=${SERVER_LOG:-$VAGEN_LOG_ROOT/navigation-ai2thor-server-${SERVER_SESSION_ID}-${SLURM_JOB_ID:-manual}.log}
export VAGEN_AI2THOR_HOME=${VAGEN_AI2THOR_HOME:-$VAGEN_ARTIFACT_ROOT/ai2thor-home/${SLURM_JOB_ID:-manual}-server}

cd "$VAGEN_REPO"
source scripts/superpod/start_local_server.sh

SERVER_HOST=$(hostname -f 2>/dev/null || hostname)
SERVER_IP=$(hostname -I 2>/dev/null | tr ' ' '\n' | awk '/^10\.23\./ {print; exit}' || true)
if [ -z "$SERVER_IP" ]; then
  SERVER_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || true)
fi
SERVER_ADVERTISE_HOST=${SERVER_ADVERTISE_HOST:-${SERVER_IP:-$SERVER_HOST}}
cat > "$SERVER_READY_FILE" <<EOF
export SERVER_SESSION_ID=$SERVER_SESSION_ID
export SERVER_JOB_ID=${SLURM_JOB_ID:-manual}
export AI2THOR_SERVER_HOST=$SERVER_HOST
export AI2THOR_SERVER_IP=${SERVER_IP:-}
export AI2THOR_SERVER_NODE=$SERVER_ADVERTISE_HOST
export SERVER_PORT=$SERVER_PORT
export ROLLOUT_BASE_URL=http://$SERVER_ADVERTISE_HOST:$SERVER_PORT
export SERVER_LOG=$SERVER_LOG
export SERVER_NAVIGATION_MAX_WORKERS=$SERVER_NAVIGATION_MAX_WORKERS
export AI2THOR_SERVER_NAVIGATION_MAX_WORKERS=$SERVER_NAVIGATION_MAX_WORKERS
export SERVER_NAVIGATION_DEVICES=$SERVER_NAVIGATION_DEVICES
EOF

echo "AI2-THOR server ready file: $SERVER_READY_FILE"
cat "$SERVER_READY_FILE"
wait "$VAGEN_SERVER_PID"
