#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/project/hf_cache
export HUGGINGFACE_HUB_CACHE=/project/hf_cache/hub
# Keep this for compatibility with current env; transformers will prefer HF_HOME.
export TRANSFORMERS_CACHE=/project/hf_cache/transformers
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0

MODEL_PATH="$(python - <<'PY'
from huggingface_hub import snapshot_download
print(snapshot_download(repo_id="Qwen/Qwen2.5-VL-3B-Instruct", cache_dir="/project/hf_cache/hub"))
PY
)"

PYTHONUNBUFFERED=1 python -m vagen.main_ppo \
  --config-path=./configs \
  --config-name=vagen_multiturn \
  data.train_files=./examples/train/navigation/train_navigation.yaml \
  data.val_files=./examples/train/navigation/val_navigation.yaml \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.agent.agent_loop_config_path=./vagen/configs/agent_no_concat.yaml \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  trainer.concat_multi_turn=False \
  algorithm.adv_estimator=no_concat_gae \
  actor_rollout_ref.actor.num_actions=8 \
  actor_rollout_ref.actor.world_state_dim=256 \
  actor_rollout_ref.actor.enable_latent_mcts=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=2 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=768 \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
  actor_rollout_ref.actor.entropy_checkpointing=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.pipeline_model_parallel_size=1 \
  actor_rollout_ref.rollout.data_parallel_size=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  data.max_prompt_length=512 \
  data.max_response_length=256 \
  actor_rollout_ref.rollout.prompt_length=512 \
  actor_rollout_ref.rollout.response_length=256 \
  actor_rollout_ref.rollout.max_model_len=512 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.20 \
  actor_rollout_ref.rollout.max_num_batched_tokens=128 \
  actor_rollout_ref.rollout.max_num_seqs=1 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.enable_prefix_caching=False \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.n=1 \
  data.train_batch_size=2 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.logger="['console']" \
  trainer.val_before_train=False \
  trainer.test_freq=5 \
  trainer.save_freq=0 \
  trainer.total_training_steps=10 \
  actor_rollout_ref.rollout.agent.num_workers=1 \
  trainer.resume_mode=disable \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.actor.optim.optimizer_impl=torchao.optim \
  actor_rollout_ref.actor.optim.optimizer=_AdamW \
  +actor_rollout_ref.actor.optim.override_optimizer_config.bf16_stochastic_round=True