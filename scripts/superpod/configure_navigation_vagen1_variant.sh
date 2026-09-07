#!/usr/bin/env bash
set -euo pipefail

set_vagen_runtime() {
  local gpu_memory_utilization="${1:-0.4}"
  local limit_mm_per_prompt="${2:-8}"
  export ROLLOUT_LIMIT_MM_PER_PROMPT="$limit_mm_per_prompt"
  export ROLLOUT_ENABLE_CHUNKED_PREFILL=False
  export ROLLOUT_ENFORCE_EAGER=False
  export ROLLOUT_FREE_CACHE_ENGINE=False
  export ROLLOUT_MAX_NUM_BATCHED_TOKENS=8192
  export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=4
  export ROLLOUT_GPU_MEMORY_UTILIZATION="$gpu_memory_utilization"
}

set_eager_runtime() {
  set_vagen_runtime "${1:-0.4}" "${2:-8}"
  export ROLLOUT_ENFORCE_EAGER=True
}

set_speed_runtime() {
  local train_batch_size="$1"
  local ppo_mini_batch_size="$2"
  local rollout_mini_batch_size="$3"
  local server_workers="$4"
  set_vagen_runtime 0.6 8
  export TRAIN_BATCH_SIZE="$train_batch_size"
  export PPO_MINI_BATCH_SIZE="$ppo_mini_batch_size"
  export ROLLOUT_MINI_BATCH_SIZE="$rollout_mini_batch_size"
  export SERVER_NAVIGATION_MAX_WORKERS="$server_workers"
  export MAX_TURNS=20
}

variant="${VAGEN1_VARIANT:-manual}"
case "$variant" in
  manual)
    ;;
  vagenrt_gpu01_limit5)
    set_vagen_runtime 0.1 5
    ;;
  vagenrt_gpu04_limit5)
    set_vagen_runtime 0.4 5
    ;;
  vagenrt_gpu06_limit5)
    set_vagen_runtime 0.6 5
    ;;
  eager_gpu04_limit5)
    set_eager_runtime 0.4 5
    ;;
  eager_free_gpu04_limit5)
    set_eager_runtime 0.4 5
    export ROLLOUT_FREE_CACHE_ENGINE=True
    ;;
  eager_chunk_gpu04_limit5)
    set_eager_runtime 0.4 5
    export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
    export ROLLOUT_MAX_NUM_BATCHED_TOKENS=10000
    ;;
  vagenrt_gpu04_limit8)
    set_vagen_runtime 0.4 8
    ;;
  vagenrt_gpu06_limit8)
    set_vagen_runtime 0.6 8
    ;;
  eager_gpu04_limit8)
    set_eager_runtime 0.4 8
    ;;
  eager_gpu06_limit8)
    set_eager_runtime 0.6 8
    ;;
  eager_tiny)
    set_eager_runtime 0.4 8
    ;;
  eager_actionpen)
    set_eager_runtime 0.4 8
    ;;
  eager_guard)
    set_eager_runtime 0.4 8
    ;;
  eager_free_gpu04_limit8)
    set_eager_runtime 0.4 8
    export ROLLOUT_FREE_CACHE_ENGINE=True
    ;;
  eager_chunk_gpu04_limit8)
    set_eager_runtime 0.4 8
    export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
    export ROLLOUT_MAX_NUM_BATCHED_TOKENS=10000
    ;;
  failed_minus_limit20_gpu06)
    export ROLLOUT_LIMIT_MM_PER_PROMPT=5
    export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
    export ROLLOUT_ENFORCE_EAGER=True
    export ROLLOUT_FREE_CACHE_ENGINE=True
    export ROLLOUT_MAX_NUM_BATCHED_TOKENS=10000
    export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=4
    export ROLLOUT_GPU_MEMORY_UTILIZATION=0.6
    ;;
  failed_minus_limit8_gpu06)
    export ROLLOUT_LIMIT_MM_PER_PROMPT=8
    export ROLLOUT_ENABLE_CHUNKED_PREFILL=True
    export ROLLOUT_ENFORCE_EAGER=True
    export ROLLOUT_FREE_CACHE_ENGINE=True
    export ROLLOUT_MAX_NUM_BATCHED_TOKENS=10000
    export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=4
    export ROLLOUT_GPU_MEMORY_UTILIZATION=0.6
    ;;
  tp1_diag_gpu06_limit5)
    set_eager_runtime 0.6 5
    export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
    ;;
  tp1_diag_gpu06_limit8)
    set_eager_runtime 0.6 8
    export ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=1
    ;;
  batch64_turn20)
    set_vagen_runtime "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
    export TRAIN_BATCH_SIZE=64
    export PPO_MINI_BATCH_SIZE=32
    export MAX_TURNS=20
    ;;
  batch128_turn20)
    set_vagen_runtime "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
    export TRAIN_BATCH_SIZE=128
    export PPO_MINI_BATCH_SIZE=32
    export MAX_TURNS=20
    ;;
  batch256_turn20)
    set_vagen_runtime "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
    export TRAIN_BATCH_SIZE=256
    export PPO_MINI_BATCH_SIZE=64
    export MAX_TURNS=20
    ;;
  batch128_turn15)
    set_vagen_runtime "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
    export TRAIN_BATCH_SIZE=128
    export PPO_MINI_BATCH_SIZE=32
    export MAX_TURNS=15
    ;;
  batch128_turn10_diag)
    set_vagen_runtime "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
    export TRAIN_BATCH_SIZE=128
    export PPO_MINI_BATCH_SIZE=32
    export MAX_TURNS=10
    ;;
  batch128_workers_sweep)
    set_vagen_runtime "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
    export TRAIN_BATCH_SIZE=128
    export PPO_MINI_BATCH_SIZE=32
    export MAX_TURNS=20
    export SERVER_NAVIGATION_MAX_WORKERS=2
    ;;
  speed_b8_rmb1_w1_turn20)
    set_speed_runtime 8 8 1 1
    ;;
  speed_b8_rmb4_w1_turn20)
    set_speed_runtime 8 8 4 1
    ;;
  speed_b8_rmb8_w1_turn20)
    set_speed_runtime 8 8 8 1
    ;;
  speed_b16_rmb4_w2_turn20)
    set_speed_runtime 16 16 4 2
    ;;
  speed_b16_rmb8_w2_turn20)
    set_speed_runtime 16 16 8 2
    ;;
  speed_b16_rmb16_w4_turn20)
    set_speed_runtime 16 16 16 4
    ;;
  speed_b32_rmb8_w4_turn20)
    set_speed_runtime 32 32 8 4
    ;;
  speed_b32_rmb16_w4_turn20)
    set_speed_runtime 32 32 16 4
    ;;
  speed_b64_rmb16_w4_turn20)
    set_speed_runtime 64 32 16 4
    ;;
  speed_b64_rmb32_w4_turn20)
    set_speed_runtime 64 32 32 4
    ;;
  turn10_defaultloss_fmt01)
    set_vagen_runtime 0.4 8
    export TRAIN_BATCH_SIZE=16
    export PPO_MINI_BATCH_SIZE=16
    export ROLLOUT_MINI_BATCH_SIZE=4
    export SERVER_NAVIGATION_MAX_WORKERS=2
    export MAX_TURNS=10
    export FORMAT_REWARD=0.1
    export LOSS_MASK_MODE=default
    ;;
  turn10_answeronly_fmt005_rmb2_w1)
    set_vagen_runtime 0.4 8
    export TRAIN_BATCH_SIZE=16
    export PPO_MINI_BATCH_SIZE=16
    export ROLLOUT_MINI_BATCH_SIZE=2
    export SERVER_NAVIGATION_MAX_WORKERS=1
    export MAX_TURNS=10
    export FORMAT_REWARD=0.05
    export LOSS_MASK_MODE=answer_only
    ;;
  turn5_answeronly_fmt005_rmb2_w1)
    set_vagen_runtime 0.4 8
    export TRAIN_BATCH_SIZE=16
    export PPO_MINI_BATCH_SIZE=16
    export ROLLOUT_MINI_BATCH_SIZE=2
    export SERVER_NAVIGATION_MAX_WORKERS=1
    export MAX_TURNS=5
    export FORMAT_REWARD=0.05
    export LOSS_MASK_MODE=answer_only
    ;;
  external_b16_rmb16_w4_turn20)
    set_speed_runtime 16 16 16 4
    ;;
  external_b32_rmb16_w4_turn20)
    set_speed_runtime 32 32 16 4
    ;;
  external_b32_rmb32_w8_turn20)
    set_speed_runtime 32 32 32 8
    ;;
  external2_b16_rmb16_w4x2)
    set_speed_runtime 16 16 16 4
    ;;
  external2_b32_rmb16_w4x2)
    set_speed_runtime 32 32 16 4
    ;;
  external2_train2x4_b16_rmb16_w4x2)
    set_speed_runtime 16 16 16 4
    ;;
  external2_train2x4_b32_rmb16_w4x2)
    set_speed_runtime 32 32 16 4
    ;;
  fmt01_defaultloss)
    set_vagen_runtime "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
    export FORMAT_REWARD=0.1
    export LOSS_MASK_MODE=default
    ;;
  fmt01_answeronly)
    set_vagen_runtime "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
    export FORMAT_REWARD=0.1
    export LOSS_MASK_MODE=answer_only
    ;;
  fmt005_answeronly)
    set_vagen_runtime "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
    export FORMAT_REWARD=0.05
    export LOSS_MASK_MODE=answer_only
    ;;
  fmt01_answeronly_strict)
    set_vagen_runtime "${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.4}"
    export FORMAT_REWARD=0.1
    export LOSS_MASK_MODE=answer_only
    export VAGEN1_STRICT_FORMAT=1
    ;;
  *)
    echo "Unknown VAGEN1_VARIANT=$variant" >&2
    exit 2
    ;;
esac
