#!/usr/bin/env bash
#SBATCH --job-name=vagen-nav-498593-infer
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --qos=normal_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=/project/peilab/hligb/vagen-navigation/logs/navigation-498593-step30-full-infer-1gpu-%j.out
#SBATCH --error=/project/peilab/hligb/vagen-navigation/logs/navigation-498593-step30-full-infer-1gpu-%j.err

set -euo pipefail

export VAGEN_REPO=${VAGEN_REPO:-/home/hligb/test_lu/VAGEN-navigation-repro-vagen1-train2x4-ffaf505}
export VAGEN_ARTIFACT_ROOT=${VAGEN_ARTIFACT_ROOT:-/project/peilab/hligb/vagen-navigation}
cd "$VAGEN_REPO"

source scripts/superpod/prepare_project_storage.sh

export VAGEN_NODE_LOCAL_ROOT=${VAGEN_NODE_LOCAL_ROOT:-/tmp/${USER:-hligb}/vn-fullinfer/${SLURM_JOB_ID:-manual}}
export VAGEN_AI2THOR_HOME=${VAGEN_LOCAL_AI2THOR_HOME:-$VAGEN_NODE_LOCAL_ROOT/a}
export RAY_TMPDIR=$VAGEN_NODE_LOCAL_ROOT/r
export TMPDIR=$VAGEN_NODE_LOCAL_ROOT/t
export TRITON_CACHE_DIR=$VAGEN_NODE_LOCAL_ROOT/tr
export TORCHINDUCTOR_CACHE_DIR=$VAGEN_NODE_LOCAL_ROOT/i
export TORCH_EXTENSIONS_DIR=$VAGEN_NODE_LOCAL_ROOT/e
mkdir -p "$VAGEN_AI2THOR_HOME" "$RAY_TMPDIR" "$TMPDIR" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

source scripts/superpod/load_modules.sh
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${VAGEN_CONDA_ENV:-vagen_nav}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}
export VLLM_DO_NOT_TRACK=${VLLM_DO_NOT_TRACK:-1}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}
export RAY_DEDUP_LOGS=${RAY_DEDUP_LOGS:-0}
unset PYTORCH_CUDA_ALLOC_CONF
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
export TORCHINDUCTOR_DISABLE=${TORCHINDUCTOR_DISABLE:-1}
export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export WANDB_MODE=${WANDB_MODE:-disabled}
export SERVER_USE_STATE_REWARD=${SERVER_USE_STATE_REWARD:-False}
export SERVER_AI2THOR_LOCAL_RELEASES=${SERVER_AI2THOR_LOCAL_RELEASES:-1}
export SERVER_PREWARM_AI2THOR=${SERVER_PREWARM_AI2THOR:-0}
export SERVER_NAVIGATION_MAX_WORKERS=${SERVER_NAVIGATION_MAX_WORKERS:-1}
export SERVER_NAVIGATION_CREATE_MAX_WORKERS=${SERVER_NAVIGATION_CREATE_MAX_WORKERS:-1}
export SERVER_NAVIGATION_DEVICES=${SERVER_NAVIGATION_DEVICES:-0}
export SERVER_PORT=${SERVER_PORT:-$((7400 + ${SLURM_JOB_ID:-0} % 500))}
export SERVER_LOG=${SERVER_LOG:-$VAGEN_LOG_ROOT/navigation-498593-step30-fullinfer-local-server-${SLURM_JOB_ID:-manual}.log}

RUN_TAG=${RUN_TAG:-498593_step30_full_$(date -u +%Y%m%dT%H%M%SZ)}
EXP=${EXP:-navigation_vagen1_guard_w1_rmb2_turn10_4h_20260730T174706Z}
CKPT=${CKPT:-$VAGEN_ARTIFACT_ROOT/checkpoints/vagen_navigation_repro/$EXP/global_step_30}
ACTOR_DIR=${ACTOR_DIR:-$CKPT/actor}
SOURCE_HF_DIR="$ACTOR_DIR/huggingface"
MERGE_ACTOR_DIR=${MERGE_ACTOR_DIR:-$VAGEN_NODE_LOCAL_ROOT/actor_for_hf_merge}
HF_DIR="$MERGE_ACTOR_DIR/huggingface"
DATA_DIR=${DATA_DIR:-$VAGEN_ARTIFACT_ROOT/data/$EXP}
CHUNK_SIZE=${CHUNK_SIZE:-64}
SERVER_MAX_WORKERS=${SERVER_MAX_WORKERS:-1}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-1200}
MAX_STEPS=${MAX_STEPS:-10}

if [ ! -d "$ACTOR_DIR" ]; then
  echo "missing actor checkpoint dir: $ACTOR_DIR" >&2
  exit 2
fi

if [ ! -f "$DATA_DIR/train.parquet" ] || [ ! -f "$DATA_DIR/test.parquet" ]; then
  echo "missing data parquet under $DATA_DIR" >&2
  exit 2
fi

if ! find "$SOURCE_HF_DIR" -maxdepth 1 \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) | grep -q .; then
  echo "merged HF weights not found under $SOURCE_HF_DIR; merging FSDP shards to node-local $HF_DIR"
  rm -rf "$MERGE_ACTOR_DIR"
  mkdir -p "$MERGE_ACTOR_DIR" "$HF_DIR"
  find "$ACTOR_DIR" -maxdepth 1 -type f \( -name 'model_world_size_*_rank_*.pt' -o -name 'extra_state_world_size_*_rank_*.pt' \) \
    -exec ln -s {} "$MERGE_ACTOR_DIR/" \;
  find "$SOURCE_HF_DIR" -maxdepth 1 -type f ! \( -name '*.safetensors' -o -name 'pytorch_model*.bin' -o -name 'model.safetensors.index.json' \) \
    -exec cp -a {} "$HF_DIR/" \;
  python /project/peilab/hligb/vagen-navigation/verl/scripts/model_merger.py --local_dir "$MERGE_ACTOR_DIR"
else
  echo "merged HF weights already exist under $SOURCE_HF_DIR"
  HF_DIR="$SOURCE_HF_DIR"
fi

MODEL_PATH="$HF_DIR"

source scripts/superpod/configure_vulkan_env.sh
source scripts/superpod/start_local_server.sh
SERVER_URL="http://127.0.0.1:$SERVER_PORT"
trap 'cleanup_vagen_server || true' EXIT

python - "$SERVER_URL" <<'PYHEALTH'
import json
import sys
import urllib.request

url = sys.argv[1].rstrip("/")
with urllib.request.urlopen(url + "/health", timeout=10) as response:
    payload = json.loads(response.read().decode("utf-8"))
if payload.get("status") != "ok":
    raise SystemExit(f"bad health payload: {payload}")
print(f"local env server healthy: {url}")
PYHEALTH

run_split() {
  local split="$1"
  local parquet="$DATA_DIR/${split}.parquet"
  local output_dir="$VAGEN_ARTIFACT_ROOT/eval/full_${RUN_TAG}_${split}_all"

  echo "===== full infer split=$split ====="
  echo "model=$MODEL_PATH"
  echo "parquet=$parquet"
  echo "server_url=$SERVER_URL"
  echo "output_dir=$output_dir"
  echo "chunk_size=$CHUNK_SIZE max_steps=$MAX_STEPS"
  echo "server_navigation_max_workers=$SERVER_NAVIGATION_MAX_WORKERS server_max_workers=$SERVER_MAX_WORKERS"

  python -m vagen.inference.navigation_chunked_eval \
    --model_path "$MODEL_PATH" \
    --val_files_path "$parquet" \
    --server_url "$SERVER_URL" \
    --output_dir "$output_dir" \
    --chunk_size "$CHUNK_SIZE" \
    --max_examples 0 \
    --max_steps "$MAX_STEPS" \
    --window_size 5 \
    --server_max_workers "$SERVER_MAX_WORKERS" \
    --server_timeout "$SERVER_TIMEOUT" \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.75}" \
    --temperature "${INFER_TEMPERATURE:-0.7}" \
    --top_p "${INFER_TOP_P:-0.95}" \
    --top_k "${INFER_TOP_K:-50}" \
    --max_tokens "${INFER_MAX_TOKENS:-256}" \
    --seed "${INFER_SEED:-0}" \
    --split "${split}"
}

run_split test
run_split train

echo "===== full infer finished: $RUN_TAG ====="
