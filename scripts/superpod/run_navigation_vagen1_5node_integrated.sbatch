#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-vagen1-5n-int
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-5node-integrated-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-5node-integrated-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
source "$VAGEN_REPO/scripts/superpod/prepare_project_storage.sh"

export ENV_NNODES=${ENV_NNODES:-2}
export TRAIN_NNODES=${TRAIN_NNODES:-3}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export ENV_GPUS_PER_NODE=${ENV_GPUS_PER_NODE:-$N_GPUS_PER_NODE}
export EXPECTED_RAY_GPUS=${EXPECTED_RAY_GPUS:-12}
export SERVER_WAIT_SECONDS=${SERVER_WAIT_SECONDS:-900}
export RAY_CLUSTER_WAIT_SECONDS=${RAY_CLUSTER_WAIT_SECONDS:-300}

mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
required_nodes=$((ENV_NNODES + TRAIN_NNODES))
if [ "${#ALL_NODES[@]}" -lt "$required_nodes" ]; then
  echo "integrated VAGEN1 job requires $required_nodes nodes, got ${#ALL_NODES[@]}: ${ALL_NODES[*]:-none}" >&2
  exit 2
fi

ENV_NODES=("${ALL_NODES[@]:0:$ENV_NNODES}")
TRAIN_NODES=("${ALL_NODES[@]:$ENV_NNODES:$TRAIN_NNODES}")
export HEAD_NODE=${HEAD_NODE:-${TRAIN_NODES[0]}}
export RAY_PORT=${RAY_PORT:-$((6379 + ${SLURM_JOB_ID:-0} % 1000))}
export HEAD_IP
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" bash -lc "hostname -I | tr ' ' '\n' | grep -m1 '^10\\.' || hostname -I | awk '{print \$1}'")
export RAY_HEAD_ADDRESS="$HEAD_IP:$RAY_PORT"
export RAY_TMP_ROOT=${RAY_TMP_ROOT:-$VAGEN_ARTIFACT_ROOT/ray/$SLURM_JOB_ID}
mkdir -p "$RAY_TMP_ROOT" "$VAGEN_LOG_ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export WANDB_MODE=${WANDB_MODE:-online}
export VAGEN1_VARIANT=${VAGEN1_VARIANT:-manual}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_vagen1_integrated_5node_answeronly_20260723}
export WANDB_TAGS=${WANDB_TAGS:-vagen,navigation,vagen1,step1,integrated,external-ai2thor,multi-server,normal,train3x4,env2x4,b96,rmb48,total60,answer-only,fmt005}
export RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
export SERVER_SESSION_ID=${SERVER_SESSION_ID:-navigation_vagen1_integrated_5node_${SLURM_JOB_ID:-manual}_${RUN_STAMP}}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-navigation_vagen1_integrated_5node_answeronly_b96_rmb48_${RUN_STAMP}}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-1}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-60}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-96}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-48}
export ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-48}
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-4}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
export ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT:-1800}
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
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-False}
export RAW_SAMPLES_TO_LOG=${RAW_SAMPLES_TO_LOG:-0}
export RAW_SAMPLES_MAX_CHARS=${RAW_SAMPLES_MAX_CHARS:-2000}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:-20}
export REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE:-False}
export ENV_CONFIG_PATH=${ENV_CONFIG_PATH:-$VAGEN_REPO/scripts/examples/vagen_base/navigation_vagen1/env_config_speed.yaml}
export SERVER_USE_STATE_REWARD=${SERVER_USE_STATE_REWARD:-False}
export SERVER_PREWARM_AI2THOR=${SERVER_PREWARM_AI2THOR:-0}
export SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-4}
export SERVER_NAVIGATION_DEVICES=${SERVER_NAVIGATION_DEVICES:-}
if [ -z "$SERVER_NAVIGATION_DEVICES" ]; then
  SERVER_NAVIGATION_DEVICES="0"
  for ((device_idx = 1; device_idx < ENV_GPUS_PER_NODE; device_idx++)); do
    SERVER_NAVIGATION_DEVICES="${SERVER_NAVIGATION_DEVICES},${device_idx}"
  done
  export SERVER_NAVIGATION_DEVICES
fi
export VAGEN_RUN_LOG=${VAGEN_RUN_LOG:-$VAGEN_LOG_ROOT/$EXPERIMENT_NAME.log}

if [ "$VAGEN1_VARIANT" != "manual" ]; then
  source "$VAGEN_REPO/scripts/superpod/configure_navigation_vagen1_variant.sh"
fi

ENV_PIDS=()
RAY_PIDS=()
SERVER_READY_FILE_ARRAY=()

cleanup_integrated_job() {
  for pid in "${RAY_PIDS[@]:-}" "${ENV_PIDS[@]:-}"; do
    if [ -n "${pid:-}" ] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  for node in "${TRAIN_NODES[@]:-}"; do
    srun --overlap --nodes=1 --ntasks=1 --nodelist="$node" bash -lc '
      source "$VAGEN_REPO/scripts/superpod/load_modules.sh" >/dev/null 2>&1 || true
      source "$(conda info --base)/etc/profile.d/conda.sh" >/dev/null 2>&1 || true
      conda activate "${VAGEN_CONDA_ENV:-vagen_nav}" >/dev/null 2>&1 || true
      ray stop --force >/dev/null 2>&1 || true
    ' >/dev/null 2>&1 || true
  done
}
trap cleanup_integrated_job EXIT

start_env_server() {
  local node="$1"
  local rank="$2"
  local port=$((7300 + (${SLURM_JOB_ID:-0} % 400) + rank))
  local session="${SERVER_SESSION_ID}_server${rank}"
  local ready_file="$VAGEN_LOG_ROOT/navigation-ai2thor-server-${session}.env"
  local server_log="$VAGEN_LOG_ROOT/navigation-ai2thor-server-${session}-${SLURM_JOB_ID:-manual}.log"
  local stdout_log="$VAGEN_LOG_ROOT/integrated-env-server-${rank}-${SLURM_JOB_ID:-manual}.out"
  local stderr_log="$VAGEN_LOG_ROOT/integrated-env-server-${rank}-${SLURM_JOB_ID:-manual}.err"

  rm -f "$ready_file"
  SERVER_READY_FILE_ARRAY+=("$ready_file")

  (
    export SERVER_SESSION_ID="$session"
    export SERVER_READY_FILE="$ready_file"
    export SERVER_PORT="$port"
    export SERVER_LOG="$server_log"
    export VAGEN_AI2THOR_HOME="$VAGEN_ARTIFACT_ROOT/ai2thor-home/${SLURM_JOB_ID:-manual}-server${rank}"
    srun --overlap --nodes=1 --ntasks=1 --nodelist="$node" --gres=gpu:$ENV_GPUS_PER_NODE --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
      bash -lc '
        set -euo pipefail
        cd "$VAGEN_REPO"
        source scripts/superpod/prepare_project_storage.sh
        source scripts/superpod/load_modules.sh
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"
        export CUDA_VISIBLE_DEVICES=${SERVER_NAVIGATION_DEVICES:-${CUDA_VISIBLE_DEVICES:-0,1,2,3}}
        export SERVER_NAVIGATION_DEVICES=${SERVER_NAVIGATION_DEVICES:-$CUDA_VISIBLE_DEVICES}
        source scripts/superpod/start_local_server.sh

        SERVER_HOST=$(hostname -f 2>/dev/null || hostname)
        SERVER_IP=$(hostname -I 2>/dev/null | tr " " "\n" | grep -m1 "^10\\.23\\." || true)
        if [ -z "$SERVER_IP" ]; then
          SERVER_IP=$(hostname -I 2>/dev/null | awk "{print \$1}" || true)
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
        echo "AI2-THOR server is ready for integrated job: $SERVER_READY_FILE"
        cat "$SERVER_READY_FILE"
        wait "$VAGEN_SERVER_PID"
      '
  ) > "$stdout_log" 2> "$stderr_log" &
  ENV_PIDS+=("$!")
}

wait_for_env_servers() {
  local ready_files_csv
  ready_files_csv=$(IFS=,; echo "${SERVER_READY_FILE_ARRAY[*]}")
  python - "$ready_files_csv" "$SERVER_WAIT_SECONDS" <<'PY'
import pathlib
import sys
import time

ready_files = [pathlib.Path(item.strip()) for item in sys.argv[1].split(",") if item.strip()]
deadline = time.time() + int(sys.argv[2])
while time.time() < deadline:
    missing = [path for path in ready_files if not path.exists() or path.stat().st_size == 0]
    if not missing:
        print("found integrated AI2-THOR server ready files:")
        for ready_file in ready_files:
            print(f"  {ready_file}")
        sys.exit(0)
    time.sleep(5)

print("timed out waiting for integrated AI2-THOR server ready files:", file=sys.stderr)
for ready_file in missing:
    print(f"  {ready_file}", file=sys.stderr)
sys.exit(1)
PY
}

collect_and_check_env_servers() {
  ROLLOUT_BASE_URLS_COLLECTED=()
  SERVER_JOB_IDS_COLLECTED=()
  SERVER_DESCRIPTIONS=()

  for ready_file in "${SERVER_READY_FILE_ARRAY[@]}"; do
    set -a
    # shellcheck disable=SC1090
    source "$ready_file"
    set +a

    if [ "${SERVER_JOB_ID:-}" != "${SLURM_JOB_ID:-manual}" ]; then
      echo "stale ready file rejected: $ready_file has SERVER_JOB_ID=${SERVER_JOB_ID:-missing}, expected ${SLURM_JOB_ID:-manual}" >&2
      exit 3
    fi

    python - "$AI2THOR_SERVER_NODE" "$SERVER_PORT" <<'PY'
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
deadline = time.time() + 300
while time.time() < deadline:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(2)
        if sock.connect_ex((host, port)) == 0:
            print(f"AI2-THOR server is reachable on {host}:{port}")
            sys.exit(0)
    time.sleep(5)

print(f"AI2-THOR server is not reachable on {host}:{port}", file=sys.stderr)
sys.exit(1)
PY

    ROLLOUT_BASE_URLS_COLLECTED+=("$ROLLOUT_BASE_URL")
    SERVER_JOB_IDS_COLLECTED+=("${SERVER_JOB_ID:-unknown}")
    SERVER_DESCRIPTIONS+=("${SERVER_JOB_ID:-unknown}@${AI2THOR_SERVER_HOST:-unknown}:${SERVER_PORT}=${ROLLOUT_BASE_URL}")
  done

  export ROLLOUT_BASE_URL=${ROLLOUT_BASE_URLS_COLLECTED[0]}
  export ROLLOUT_BASE_URLS=$(IFS=,; echo "${ROLLOUT_BASE_URLS_COLLECTED[*]}")
  export SERVER_JOB_IDS=$(IFS=,; echo "${SERVER_JOB_IDS_COLLECTED[*]}")
  export SERVER_READY_FILES=$(IFS=,; echo "${SERVER_READY_FILE_ARRAY[*]}")
}

start_ray_head() {
  srun --overlap --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" --gres=gpu:4 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
    bash -lc '
      set -euo pipefail
      cd "$VAGEN_REPO"
      source scripts/superpod/load_modules.sh
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"
      export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
      export RAY_TMPDIR="$RAY_TMP_ROOT/head"
      mkdir -p "$RAY_TMPDIR"
      ray stop --force || true
      ray start --head --node-ip-address="$HEAD_IP" --port="$RAY_PORT" --num-cpus="${SLURM_CPUS_PER_TASK:-64}" --num-gpus="$N_GPUS_PER_NODE" --temp-dir="$RAY_TMPDIR" --block
    ' > "$VAGEN_LOG_ROOT/ray-head-${SLURM_JOB_ID}.log" 2>&1 &
  RAY_PIDS+=("$!")
}

start_ray_worker() {
  local node="$1"
  local rank="$2"
  srun --overlap --nodes=1 --ntasks=1 --nodelist="$node" --gres=gpu:4 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
    bash -lc '
      set -euo pipefail
      cd "$VAGEN_REPO"
      source scripts/superpod/load_modules.sh
      source "$(conda info --base)/etc/profile.d/conda.sh"
      conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"
      export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
      export RAY_TMPDIR="$RAY_TMP_ROOT/worker-'$rank'"
      mkdir -p "$RAY_TMPDIR"
      ray stop --force || true
      ray start --address="$RAY_HEAD_ADDRESS" --num-cpus="${SLURM_CPUS_PER_TASK:-64}" --num-gpus="$N_GPUS_PER_NODE" --temp-dir="$RAY_TMPDIR" --block
    ' > "$VAGEN_LOG_ROOT/ray-worker-${rank}-${SLURM_JOB_ID}.log" 2>&1 &
  RAY_PIDS+=("$!")
}

wait_for_ray_cluster() {
  source "$VAGEN_REPO/scripts/superpod/load_modules.sh"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"

  python - <<'PY'
import os
import sys
import time

import ray

address = os.environ["RAY_HEAD_ADDRESS"]
expected_gpus = float(os.environ.get("EXPECTED_RAY_GPUS", "12"))
deadline = time.time() + int(os.environ.get("RAY_CLUSTER_WAIT_SECONDS", "300"))
last_error = None

while time.time() < deadline:
    try:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(address=address, ignore_reinit_error=True)
        resources = ray.cluster_resources()
        gpus = float(resources.get("GPU", 0))
        alive_nodes = sum(1 for node in ray.nodes() if node.get("Alive"))
        print(f"Ray cluster resources={resources} alive_nodes={alive_nodes}")
        if gpus >= expected_gpus:
            ray.shutdown()
            sys.exit(0)
    except Exception as exc:  # noqa: BLE001
        last_error = exc
        print(f"waiting for Ray cluster at {address}: {exc}", file=sys.stderr)
    time.sleep(5)

if ray.is_initialized():
    ray.shutdown()
print(f"Ray cluster did not reach {expected_gpus} GPUs before timeout; last_error={last_error}", file=sys.stderr)
sys.exit(1)
PY
}

run_training_driver() {
  srun --overlap --nodes=1 --ntasks=1 --nodelist="$HEAD_NODE" bash -lc '
    set -euo pipefail
    cd "$VAGEN_REPO"
    source scripts/superpod/load_modules.sh
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"
    source scripts/superpod/configure_vulkan_env.sh
    export VAGEN_GIT_COMMIT=${VAGEN_GIT_COMMIT:-$(git rev-parse HEAD)}
    export RAY_ADDRESS="$RAY_HEAD_ADDRESS"
    export RAY_INIT_ADDRESS="$RAY_HEAD_ADDRESS"

    bash scripts/examples/vagen_base/navigation_vagen1/run.sh
    bash scripts/superpod/write_run_summary.sh "$EXPERIMENT_NAME" \
      "VAGEN_GIT_COMMIT=$VAGEN_GIT_COMMIT VAGEN1_VARIANT=$VAGEN1_VARIANT WANDB_MODE=$WANDB_MODE EXPERIMENT_NAME=$EXPERIMENT_NAME SERVER_SESSION_ID=$SERVER_SESSION_ID SERVER_JOB_IDS=${SERVER_JOB_IDS:-unknown} SERVER_READY_FILES=${SERVER_READY_FILES:-unknown} ROLLOUT_BASE_URL=$ROLLOUT_BASE_URL ROLLOUT_BASE_URLS=$ROLLOUT_BASE_URLS ENV_NNODES=$ENV_NNODES TRAIN_NNODES=$TRAIN_NNODES N_GPUS_PER_NODE=$N_GPUS_PER_NODE RAY_HEAD_ADDRESS=$RAY_HEAD_ADDRESS TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS SAVE_FREQ=$SAVE_FREQ TEST_FREQ=$TEST_FREQ VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN FINAL_VAL_AFTER_TRAIN=$FINAL_VAL_AFTER_TRAIN TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE ROLLOUT_MINI_BATCH_SIZE=$ROLLOUT_MINI_BATCH_SIZE ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION ROLLOUT_LIMIT_MM_PER_PROMPT=$ROLLOUT_LIMIT_MM_PER_PROMPT ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE ROLLOUT_ENFORCE_EAGER=$ROLLOUT_ENFORCE_EAGER ROLLOUT_ENABLE_CHUNKED_PREFILL=$ROLLOUT_ENABLE_CHUNKED_PREFILL ROLLOUT_FREE_CACHE_ENGINE=$ROLLOUT_FREE_CACHE_ENGINE ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS FORMAT_REWARD=$FORMAT_REWARD LOSS_MASK_MODE=$LOSS_MASK_MODE MAX_TURNS=$MAX_TURNS ROLLOUT_WINDOW_SIZE=$ROLLOUT_WINDOW_SIZE UPDATE_WINDOW_SIZE=$UPDATE_WINDOW_SIZE MAX_TRAJECTORY_LENGTH=$MAX_TRAJECTORY_LENGTH ROLLOUT_MAX_TRAJECTORY_LENGTH=$ROLLOUT_MAX_TRAJECTORY_LENGTH ENV_CONFIG_PATH=$ENV_CONFIG_PATH bash scripts/examples/vagen_base/navigation_vagen1/run.sh"
  '
}

echo "integrated VAGEN1 allocation all_nodes=${ALL_NODES[*]}"
echo "integrated VAGEN1 env_nodes=${ENV_NODES[*]} train_nodes=${TRAIN_NODES[*]}"
echo "integrated VAGEN1 ray head=$HEAD_NODE head_ip=$HEAD_IP ray=$RAY_HEAD_ADDRESS expected_gpus=$EXPECTED_RAY_GPUS"
echo "integrated VAGEN1 run experiment=$EXPERIMENT_NAME group=$WANDB_RUN_GROUP train_batch=$TRAIN_BATCH_SIZE ppo_mini=$PPO_MINI_BATCH_SIZE rollout_mini=$ROLLOUT_MINI_BATCH_SIZE steps=$TOTAL_TRAINING_STEPS test_freq=$TEST_FREQ env_gpus=$ENV_GPUS_PER_NODE env_devices=$SERVER_NAVIGATION_DEVICES raw_samples=$RAW_SAMPLES_TO_LOG"

env_rank=0
for node in "${ENV_NODES[@]}"; do
  env_rank=$((env_rank + 1))
  start_env_server "$node" "$env_rank"
done

wait_for_env_servers
collect_and_check_env_servers
echo "integrated AI2-THOR servers=$(IFS=';'; echo "${SERVER_DESCRIPTIONS[*]}")"
echo "integrated AI2-THOR base_url=$ROLLOUT_BASE_URL base_urls=$ROLLOUT_BASE_URLS"

start_ray_head
sleep 20
worker_rank=0
for node in "${TRAIN_NODES[@]:1}"; do
  worker_rank=$((worker_rank + 1))
  start_ray_worker "$node" "$worker_rank"
done
wait_for_ray_cluster
run_training_driver
