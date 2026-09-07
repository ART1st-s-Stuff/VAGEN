#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-vagen1-ext
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=384G
#SBATCH --time=24:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-external-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-vagen1-external-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
source "$VAGEN_REPO/scripts/superpod/prepare_project_storage.sh"

export SERVER_SESSION_ID=${SERVER_SESSION_ID:?set SERVER_SESSION_ID to match run_navigation_vagen1_ai2thor_server.sbatch}
export SERVER_READY_FILE=${SERVER_READY_FILE:-$VAGEN_LOG_ROOT/navigation-ai2thor-server-${SERVER_SESSION_ID}.env}
export SERVER_READY_FILES=${SERVER_READY_FILES:-$SERVER_READY_FILE}
export SERVER_WAIT_SECONDS=${SERVER_WAIT_SECONDS:-7200}

python - "$SERVER_READY_FILES" "$SERVER_WAIT_SECONDS" <<'PY'
import pathlib
import sys
import time

ready_files = [pathlib.Path(item.strip()) for item in sys.argv[1].split(",") if item.strip()]
deadline = time.time() + int(sys.argv[2])
while time.time() < deadline:
    missing = [path for path in ready_files if not path.exists() or path.stat().st_size == 0]
    if not missing:
        print("found AI2-THOR server ready files:")
        for ready_file in ready_files:
            print(f"  {ready_file}")
        sys.exit(0)
    time.sleep(10)
print("timed out waiting for AI2-THOR server ready files:", file=sys.stderr)
for ready_file in missing:
    print(f"  {ready_file}", file=sys.stderr)
sys.exit(1)
PY

IFS=',' read -r -a SERVER_READY_FILE_ARRAY <<< "$SERVER_READY_FILES"
ROLLOUT_BASE_URLS_COLLECTED=()
SERVER_JOB_IDS_COLLECTED=()
SERVER_DESCRIPTIONS=()

for ready_file in "${SERVER_READY_FILE_ARRAY[@]}"; do
  ready_file="$(echo "$ready_file" | xargs)"
  set -a
  # shellcheck disable=SC1090
  source "$ready_file"
  set +a

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

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export WANDB_MODE=${WANDB_MODE:-online}
export VAGEN1_VARIANT=${VAGEN1_VARIANT:-manual}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-navigation_vagen1_external_env_debug5_20260722}
export WANDB_TAGS=${WANDB_TAGS:-vagen,navigation,vagen1,single-action,step1,4gpu,normal,external-ai2thor,debug5}
export RUN_STAMP=${RUN_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-navigation_vagen1_${VAGEN1_VARIANT}_4gpu_external_ai2thor_${RUN_STAMP}}
export WANDB_NAME=${WANDB_NAME:-$EXPERIMENT_NAME}
export FORCE_GEN_DATA=${FORCE_GEN_DATA:-1}
export TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-6}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-1}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
export N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-4}
export ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.6}
export ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-16}
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
export FORMAT_REWARD=${FORMAT_REWARD:-0.1}
export LOSS_MASK_MODE=${LOSS_MASK_MODE:-default}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-False}
export SAVE_FREQ=${SAVE_FREQ:-20}
export TEST_FREQ=${TEST_FREQ:-5}
export REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE:-False}
export ENV_CONFIG_PATH=${ENV_CONFIG_PATH:-$VAGEN_REPO/scripts/examples/vagen_base/navigation_vagen1/env_config_speed.yaml}
export VAGEN_RUN_LOG=${VAGEN_RUN_LOG:-$VAGEN_LOG_ROOT/$EXPERIMENT_NAME.log}

if [ "$VAGEN1_VARIANT" != "manual" ]; then
  source "$VAGEN_REPO/scripts/superpod/configure_navigation_vagen1_variant.sh"
fi

source "$VAGEN_REPO/scripts/superpod/load_modules.sh"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"
source "$VAGEN_REPO/scripts/superpod/configure_vulkan_env.sh"

echo "external AI2-THOR server session=$SERVER_SESSION_ID servers=$(IFS=';'; echo "${SERVER_DESCRIPTIONS[*]}")"
echo "external AI2-THOR base_url=$ROLLOUT_BASE_URL base_urls=$ROLLOUT_BASE_URLS"
echo "external VAGEN1 variant: $VAGEN1_VARIANT train_batch=$TRAIN_BATCH_SIZE ppo_mini=$PPO_MINI_BATCH_SIZE rollout_mini=$ROLLOUT_MINI_BATCH_SIZE server_workers=${AI2THOR_SERVER_NAVIGATION_MAX_WORKERS:-${SERVER_NAVIGATION_MAX_WORKERS:-unknown}}"

cd "$VAGEN_REPO"
export VAGEN_GIT_COMMIT=${VAGEN_GIT_COMMIT:-$(git rev-parse HEAD)}
bash scripts/examples/vagen_base/navigation_vagen1/run.sh
bash scripts/superpod/write_run_summary.sh "$EXPERIMENT_NAME" \
  "VAGEN_GIT_COMMIT=$VAGEN_GIT_COMMIT VAGEN1_VARIANT=$VAGEN1_VARIANT WANDB_MODE=$WANDB_MODE EXPERIMENT_NAME=$EXPERIMENT_NAME SERVER_SESSION_ID=$SERVER_SESSION_ID SERVER_JOB_IDS=${SERVER_JOB_IDS:-unknown} ROLLOUT_BASE_URL=$ROLLOUT_BASE_URL ROLLOUT_BASE_URLS=$ROLLOUT_BASE_URLS N_GPUS_PER_NODE=$N_GPUS_PER_NODE TOTAL_TRAINING_STEPS=$TOTAL_TRAINING_STEPS SAVE_FREQ=$SAVE_FREQ TEST_FREQ=$TEST_FREQ VAL_BEFORE_TRAIN=$VAL_BEFORE_TRAIN FINAL_VAL_AFTER_TRAIN=$FINAL_VAL_AFTER_TRAIN TRAIN_BATCH_SIZE=$TRAIN_BATCH_SIZE PPO_MINI_BATCH_SIZE=$PPO_MINI_BATCH_SIZE ROLLOUT_MINI_BATCH_SIZE=$ROLLOUT_MINI_BATCH_SIZE ROLLOUT_GPU_MEMORY_UTILIZATION=$ROLLOUT_GPU_MEMORY_UTILIZATION ROLLOUT_LIMIT_MM_PER_PROMPT=$ROLLOUT_LIMIT_MM_PER_PROMPT ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE ROLLOUT_ENFORCE_EAGER=$ROLLOUT_ENFORCE_EAGER ROLLOUT_ENABLE_CHUNKED_PREFILL=$ROLLOUT_ENABLE_CHUNKED_PREFILL ROLLOUT_FREE_CACHE_ENGINE=$ROLLOUT_FREE_CACHE_ENGINE ROLLOUT_MAX_NUM_BATCHED_TOKENS=$ROLLOUT_MAX_NUM_BATCHED_TOKENS FORMAT_REWARD=$FORMAT_REWARD LOSS_MASK_MODE=$LOSS_MASK_MODE MAX_TURNS=$MAX_TURNS ROLLOUT_WINDOW_SIZE=$ROLLOUT_WINDOW_SIZE UPDATE_WINDOW_SIZE=$UPDATE_WINDOW_SIZE MAX_TRAJECTORY_LENGTH=$MAX_TRAJECTORY_LENGTH ROLLOUT_MAX_TRAJECTORY_LENGTH=$ROLLOUT_MAX_TRAJECTORY_LENGTH ENV_CONFIG_PATH=$ENV_CONFIG_PATH bash scripts/examples/vagen_base/navigation_vagen1/run.sh"
