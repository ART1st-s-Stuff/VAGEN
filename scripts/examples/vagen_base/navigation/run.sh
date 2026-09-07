#!/usr/bin/env bash
set -euo pipefail
set -x


export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONHASHSEED=0

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

# Extract experiment name from the path
# This will take the last 3 parts of the path: format/sokoban/free_think
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$(echo "$SCRIPT_DIR" | rev | cut -d'/' -f1-3 | rev | tr '/' '-')}
ENV_CONFIG_PATH=${ENV_CONFIG_PATH:-$SCRIPT_DIR/env_config.yaml}
DATA_ROOT=${VAGEN_DATA_ROOT:-$REPO_ROOT/data}
FORMAT_REWARD=${FORMAT_REWARD:-}
LOSS_MASK_MODE=${LOSS_MASK_MODE:-default}
VAGEN_GIT_COMMIT=${VAGEN_GIT_COMMIT:-unknown}
VAGEN1_VARIANT=${VAGEN1_VARIANT:-manual}
TRAIN_FILE="$DATA_ROOT/$EXPERIMENT_NAME/train.parquet"
TEST_FILE="$DATA_ROOT/$EXPERIMENT_NAME/test.parquet"
PROJECT_NAME=${VAGEN_PROJECT_NAME:-vagen_navigation_repro}
CHECKPOINT_ROOT=${VAGEN_CHECKPOINT_ROOT:-${VAGEN_ARTIFACT_ROOT:-$REPO_ROOT}/checkpoints}
CHECKPOINT_DIR="$CHECKPOINT_ROOT/$PROJECT_NAME/$EXPERIMENT_NAME"
LOG_ROOT=${VAGEN_LOG_ROOT:-${VAGEN_ARTIFACT_ROOT:-$REPO_ROOT}/logs}
RUN_LOG=${VAGEN_RUN_LOG:-$LOG_ROOT/$EXPERIMENT_NAME.log}
SERVER_PORT=${SERVER_PORT:-5000}
ROLLOUT_BASE_URL=${ROLLOUT_BASE_URL:-http://localhost:$SERVER_PORT}
ROLLOUT_BASE_URLS=${ROLLOUT_BASE_URLS:-}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
TRAIN_NNODES=${TRAIN_NNODES:-1}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-4}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.1}
ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-64}
ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT:-500}
ROLLOUT_LIMIT_MM_PER_PROMPT=${ROLLOUT_LIMIT_MM_PER_PROMPT:-5}
ROLLOUT_ENABLE_CHUNKED_PREFILL=${ROLLOUT_ENABLE_CHUNKED_PREFILL:-False}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
ROLLOUT_FREE_CACHE_ENGINE=${ROLLOUT_FREE_CACHE_ENGINE:-False}
ROLLOUT_MAX_NUM_BATCHED_TOKENS=${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}
MAX_TRAJECTORY_LENGTH=${MAX_TRAJECTORY_LENGTH:-5000}
ROLLOUT_MAX_TRAJECTORY_LENGTH=${ROLLOUT_MAX_TRAJECTORY_LENGTH:-$MAX_TRAJECTORY_LENGTH}
MAX_TURNS=${MAX_TURNS:-5}
ROLLOUT_WINDOW_SIZE=${ROLLOUT_WINDOW_SIZE:-5}
UPDATE_WINDOW_SIZE=${UPDATE_WINDOW_SIZE:-null}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
FINAL_VAL_AFTER_TRAIN=${FINAL_VAL_AFTER_TRAIN:-True}
RAW_SAMPLES_TO_LOG=${RAW_SAMPLES_TO_LOG:-0}
RAW_SAMPLES_MAX_CHARS=${RAW_SAMPLES_MAX_CHARS:-2000}
SAVE_FREQ=${SAVE_FREQ:-150}
TEST_FREQ=${TEST_FREQ:-20}
REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE:-False}
SAVE_CRITIC_CKPT=${SAVE_CRITIC_CKPT:-True}
SAVE_OPTIMIZER_CKPT=${SAVE_OPTIMIZER_CKPT:-True}
export VERL_SAVE_OPTIMIZER_CKPT="$SAVE_OPTIMIZER_CKPT"
FORCE_GEN_DATA=${FORCE_GEN_DATA:-0}
FORCE_GEN_ARGS=()
if [ "$FORCE_GEN_DATA" = "1" ]; then
    FORCE_GEN_ARGS=(--force_gen)
fi
RAY_INIT_ADDRESS=${RAY_INIT_ADDRESS:-${RAY_ADDRESS:-}}
export RAY_INIT_ADDRESS

echo "Experiment name: $EXPERIMENT_NAME"
# run 
# python -m vagen.server.server server.port=5000
# in a tmux session first
cd "$REPO_ROOT"
mkdir -p "$DATA_ROOT/$EXPERIMENT_NAME" "$CHECKPOINT_DIR" "$LOG_ROOT"

if [ -n "$FORMAT_REWARD" ]; then
    MATERIALIZED_ENV_CONFIG_PATH="$DATA_ROOT/$EXPERIMENT_NAME/env_config.format_reward_${FORMAT_REWARD}.yaml"
    python - "$ENV_CONFIG_PATH" "$MATERIALIZED_ENV_CONFIG_PATH" "$FORMAT_REWARD" <<'PY'
import sys
import yaml

source_path, target_path, format_reward = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(source_path, "r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)
for env_spec in config.values():
    env_spec.setdefault("env_config", {})["format_reward"] = format_reward
with open(target_path, "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
PY
    ENV_CONFIG_PATH="$MATERIALIZED_ENV_CONFIG_PATH"
fi

python -m vagen.env.create_dataset \
    --yaml_path "$ENV_CONFIG_PATH" \
    --train_path "$TRAIN_FILE" \
    --test_path "$TEST_FILE" \
    "${FORCE_GEN_ARGS[@]}"

# max_trajectory_length = max_prompt_length + max_response_length

python3 -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=masked_gae \
    algorithm.high_level_gamma=0.95 \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    data.max_trajectory_length=$MAX_TRAJECTORY_LENGTH \
    data.image_key=images \
    data.truncation=left \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=mse \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=$ROLLOUT_GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.enable_chunked_prefill=$ROLLOUT_ENABLE_CHUNKED_PREFILL \
    actor_rollout_ref.rollout.enforce_eager=$ROLLOUT_ENFORCE_EAGER \
    actor_rollout_ref.rollout.free_cache_engine=$ROLLOUT_FREE_CACHE_ENGINE \
    actor_rollout_ref.rollout.max_num_batched_tokens=$ROLLOUT_MAX_NUM_BATCHED_TOKENS \
    actor_rollout_ref.rollout.max_trajectory_length=$ROLLOUT_MAX_TRAJECTORY_LENGTH \
    actor_rollout_ref.rollout.max_model_len=$ROLLOUT_MAX_TRAJECTORY_LENGTH \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.limit_mm_per_prompt=$ROLLOUT_LIMIT_MM_PER_PROMPT \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.temperature=0.7 \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.model.path=Qwen/Qwen2.5-VL-3B-Instruct \
    critic.model.enable_gradient_checkpointing=True \
    critic.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    critic.ppo_micro_batch_size_per_gpu=1 \
    critic.model.fsdp_config.param_offload=False \
    critic.model.fsdp_config.optimizer_offload=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    +trainer.git_commit="$VAGEN_GIT_COMMIT" \
    +trainer.vagen1_variant="$VAGEN1_VARIANT" \
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=$TRAIN_NNODES \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.remove_previous_ckpt_in_save=$REMOVE_PREVIOUS_CKPT_IN_SAVE \
    trainer.save_critic_checkpoint=$SAVE_CRITIC_CKPT \
    rollout_manager.max_turns=$MAX_TURNS \
    rollout_manager.window_size=$ROLLOUT_WINDOW_SIZE \
    +rollout_manager.update_window_size=$UPDATE_WINDOW_SIZE \
    rollout_manager.use_multi_turn_reward=False \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    rollout_manager.loss_mask_mode=$LOSS_MASK_MODE \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.final_val_after_train=$FINAL_VAL_AFTER_TRAIN \
    trainer.val_generations_to_log_to_wandb=8 \
    trainer.raw_samples_to_log=$RAW_SAMPLES_TO_LOG \
    trainer.raw_samples_max_chars=$RAW_SAMPLES_MAX_CHARS \
    rollout_manager.n_trajectory=1 \
    rollout_manager.use_service=True \
    rollout_manager.timeout=$ROLLOUT_TIMEOUT \
    +rollout_manager.mini_batch_size=$ROLLOUT_MINI_BATCH_SIZE \
    rollout_manager.base_url="$ROLLOUT_BASE_URL" \
    rollout_manager.base_urls="$ROLLOUT_BASE_URLS" \
    2>&1 | tee "$RUN_LOG"
