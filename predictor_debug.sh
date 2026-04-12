#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
cd "$REPO_ROOT"

export HF_HOME="${HF_HOME:-/project/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
# Keep this for compatibility with current env; transformers will prefer HF_HOME.
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export MODEL_REPO_ID=Qwen/Qwen2.5-VL-3B-Instruct

: "${MODEL_REPO_ID:=Qwen/Qwen2.5-VL-3B-Instruct}"
: "${MODEL_PATH:=}"
: "${TRAIN_FILE:=./examples/train/navigation/train_navigation.yaml}"
: "${VAL_FILE:=./examples/train/navigation/val_navigation.yaml}"
: "${AGENT_LOOP_CONFIG:=./vagen/configs/agent_no_concat.yaml}"
: "${TRAIN_STEPS:=1}"
: "${TRAIN_BATCH_SIZE:=2}"
: "${PPO_MINI_BATCH_SIZE:=2}"
: "${PPO_MICRO_BATCH_SIZE:=1}"
: "${NUM_GPUS:=2}"
: "${NUM_NODES:=1}"
: "${NUM_ACTIONS:=8}"
: "${WORLD_STATE_DIM:=256}"
: "${MAX_PROMPT_LENGTH:=1024}"
: "${MAX_RESPONSE_LENGTH:=128}"
: "${ROLLOUT_GPU_MEMORY_UTILIZATION:=0.20}"

if [[ -z "$MODEL_PATH" ]]; then
  MODEL_PATH="$(python - <<'PY'
from huggingface_hub import snapshot_download
import os

repo_id = os.environ["MODEL_REPO_ID"]
cache_dir = os.environ["HUGGINGFACE_HUB_CACHE"]
print(snapshot_download(repo_id=repo_id, cache_dir=cache_dir))
PY
)"
fi

PYTHONUNBUFFERED=1 python -m vagen.main_ppo \
  --config-path=./configs \
  --config-name=vagen_multiturn \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  trainer.concat_multi_turn=False \
  algorithm.adv_estimator=no_concat_gae \
  actor_rollout_ref.actor.num_actions="$NUM_ACTIONS" \
  actor_rollout_ref.actor.world_state_dim="$WORLD_STATE_DIM" \
  actor_rollout_ref.actor.enable_latent_mcts=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=512 \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.entropy_checkpointing=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.prompt_length="$MAX_PROMPT_LENGTH" \
  actor_rollout_ref.rollout.response_length="$MAX_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.max_model_len="$MAX_PROMPT_LENGTH" \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.max_num_batched_tokens=128 \
  actor_rollout_ref.rollout.max_num_seqs=1 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.n=1 \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  trainer.nnodes="$NUM_NODES" \
  trainer.logger="['console']" \
  trainer.val_before_train=False \
  trainer.test_freq=1 \
  trainer.save_freq=0 \
  trainer.total_training_steps="$TRAIN_STEPS" \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  trainer.resume_mode=disable \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16 \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bf16 \
  +actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32 \
  actor_rollout_ref.actor.optim.optimizer_impl=torch.optim \
  actor_rollout_ref.actor.optim.optimizer=AdamW \
  +actor_rollout_ref.actor.optim.override_optimizer_config.foreach=False
