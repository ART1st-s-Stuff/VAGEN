#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-vagen1-split
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-split-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-split-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
source "$VAGEN_REPO/scripts/superpod/prepare_project_storage.sh"

mapfile -t JOB_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
if [ "${#JOB_NODES[@]}" -lt 2 ]; then
  echo "split server job requires 2 nodes, got ${#JOB_NODES[@]}: ${JOB_NODES[*]:-none}" >&2
  exit 2
fi

export TRAIN_NODE=${TRAIN_NODE:-${JOB_NODES[0]}}
export AI2THOR_SERVER_NODE=${AI2THOR_SERVER_NODE:-${JOB_NODES[1]}}
export SERVER_PORT=${SERVER_PORT:-$((6000 + ${SLURM_JOB_ID:-0} % 1000))}
export ROLLOUT_BASE_URL=${ROLLOUT_BASE_URL:-http://$AI2THOR_SERVER_NODE:$SERVER_PORT}
export SERVER_LOG=${SERVER_LOG:-$VAGEN_LOG_ROOT/navigation-split-ai2thor-server-${SLURM_JOB_ID}.log}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_vagen1_split_server_20260721}
export WANDB_TAGS=${WANDB_TAGS:-vagen,navigation,vagen1,single-action,step1,4gpu,normal,split-ai2thor}
export SERVER_USE_STATE_REWARD=${SERVER_USE_STATE_REWARD:-False}
export SERVER_PREWARM_AI2THOR=${SERVER_PREWARM_AI2THOR:-0}
export SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-4}
export SERVER_NAVIGATION_DEVICES=${SERVER_NAVIGATION_DEVICES:-0,1,2,3}
export RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-navigation_vagen1_4gpu_split_ai2thor_${RUN_STAMP}}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-1}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-4}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}
export ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-4}
export ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT:-1800}
export ROLLOUT_LIMIT_MM_PER_PROMPT=${ROLLOUT_LIMIT_MM_PER_PROMPT:-8}
export MAX_TURNS=${MAX_TURNS:-20}
export ROLLOUT_WINDOW_SIZE=${ROLLOUT_WINDOW_SIZE:-5}
export UPDATE_WINDOW_SIZE=${UPDATE_WINDOW_SIZE:-5}
export MAX_TRAJECTORY_LENGTH=${MAX_TRAJECTORY_LENGTH:-16000}
export ROLLOUT_MAX_TRAJECTORY_LENGTH=${ROLLOUT_MAX_TRAJECTORY_LENGTH:-5000}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:-20}
export REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE:-False}
export ENV_CONFIG_PATH=${ENV_CONFIG_PATH:-$VAGEN_REPO/scripts/examples/vagen_base/navigation_vagen1/env_config.yaml}
export VAGEN_RUN_LOG=${VAGEN_RUN_LOG:-$VAGEN_LOG_ROOT/$EXPERIMENT_NAME.log}

echo "split nodes: train=$TRAIN_NODE ai2thor_server=$AI2THOR_SERVER_NODE port=$SERVER_PORT base_url=$ROLLOUT_BASE_URL"
echo "logs: run=$VAGEN_RUN_LOG server=$SERVER_LOG"

srun --nodes=1 --ntasks=1 --nodelist="$AI2THOR_SERVER_NODE" --gres=gpu:4 --cpus-per-task=64 --exclusive \
  bash -lc '
    set -euo pipefail
    cd "$VAGEN_REPO"
    source scripts/superpod/prepare_project_storage.sh
    source scripts/superpod/load_modules.sh
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"
    export VAGEN_AI2THOR_HOME="$VAGEN_ARTIFACT_ROOT/ai2thor-home/${SLURM_JOB_ID}-server"
    source scripts/superpod/start_local_server.sh
    echo "AI2-THOR server ready on $(hostname):$SERVER_PORT pid=$VAGEN_SERVER_PID"
    wait "$VAGEN_SERVER_PID"
  ' &
SERVER_SRUN_PID=$!

cleanup_split_server() {
  if kill -0 "$SERVER_SRUN_PID" 2>/dev/null; then
    kill "$SERVER_SRUN_PID" 2>/dev/null || true
    wait "$SERVER_SRUN_PID" 2>/dev/null || true
  fi
}
trap cleanup_split_server EXIT

python - "$AI2THOR_SERVER_NODE" "$SERVER_PORT" "$SERVER_SRUN_PID" "$SERVER_LOG" <<'PY'
import os
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
server_pid = int(sys.argv[3])
log_path = sys.argv[4]
deadline = time.time() + 300

while time.time() < deadline:
    try:
        os.kill(server_pid, 0)
    except OSError:
        print(f"server srun exited before {host}:{port} became available", file=sys.stderr)
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
                print(handle.read()[-4000:], file=sys.stderr)
        sys.exit(1)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        if sock.connect_ex((host, port)) == 0:
            print(f"navigation server is available on {host}:{port}")
            sys.exit(0)
    time.sleep(3)

print(f"timed out waiting for navigation server on {host}:{port}", file=sys.stderr)
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8", errors="replace") as handle:
        print(handle.read()[-4000:], file=sys.stderr)
sys.exit(1)
PY

set +e
srun --nodes=1 --ntasks=1 --nodelist="$TRAIN_NODE" --gres=gpu:4 --cpus-per-task=64 --exclusive \
  bash -lc '
    set -euo pipefail
    cd "$VAGEN_REPO"
    source scripts/superpod/prepare_project_storage.sh
    source scripts/superpod/load_modules.sh
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"
    source scripts/superpod/configure_vulkan_env.sh
    bash scripts/examples/vagen_base/navigation_vagen1/run.sh
    bash scripts/superpod/write_run_summary.sh "$EXPERIMENT_NAME" \
      "WANDB_MODE=$WANDB_MODE EXPERIMENT_NAME=$EXPERIMENT_NAME TRAIN_NODE=$TRAIN_NODE AI2THOR_SERVER_NODE=$AI2THOR_SERVER_NODE ROLLOUT_BASE_URL=$ROLLOUT_BASE_URL N_GPUS_PER_NODE=$N_GPUS_PER_NODE TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS SAVE_FREQ=$SAVE_FREQ TEST_FREQ=$TEST_FREQ VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN SERVER_NAVIGATION_MAX_WORKERS=$SERVER_NAVIGATION_MAX_WORKERS ROLLOUT_MINI_BATCH_SIZE=$ROLLOUT_MINI_BATCH_SIZE ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION ROLLOUT_LIMIT_MM_PER_PROMPT=$ROLLOUT_LIMIT_MM_PER_PROMPT MAX_TURNS=$MAX_TURNS ROLLOUT_WINDOW_SIZE=$ROLLOUT_WINDOW_SIZE UPDATE_WINDOW_SIZE=$UPDATE_WINDOW_SIZE MAX_TRAJECTORY_LENGTH=$MAX_TRAJECTORY_LENGTH ROLLOUT_MAX_TRAJECTORY_LENGTH=$ROLLOUT_MAX_TRAJECTORY_LENGTH ENV_CONFIG_PATH=$ENV_CONFIG_PATH bash scripts/examples/vagen_base/navigation_vagen1/run.sh"
  '
TRAIN_EXIT=$?
set -e

exit "$TRAIN_EXIT"
