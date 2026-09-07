#!/usr/bin/env bash
set -euo pipefail
set -x

export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-XFORMERS}
export PYTHONHASHSEED=${PYTHONHASHSEED:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-$(echo "$SCRIPT_DIR" | rev | cut -d'/' -f1-3 | rev | tr '/' '-')}
ENV_CONFIG_PATH=${ENV_CONFIG_PATH:-$SCRIPT_DIR/env_config.yaml}
DATA_ROOT=${VAGEN_DATA_ROOT:-$REPO_ROOT/data}
TRAIN_FILE="$DATA_ROOT/$EXPERIMENT_NAME/train.parquet"
TEST_FILE="$DATA_ROOT/$EXPERIMENT_NAME/test.parquet"
PROJECT_NAME=${VAGEN_PROJECT_NAME:-vagen_navigation_repro}
CHECKPOINT_ROOT=${VAGEN_CHECKPOINT_ROOT:-${VAGEN_ARTIFACT_ROOT:-$REPO_ROOT}/checkpoints}
CHECKPOINT_DIR="$CHECKPOINT_ROOT/$PROJECT_NAME/$EXPERIMENT_NAME"
LOG_ROOT=${VAGEN_LOG_ROOT:-${VAGEN_ARTIFACT_ROOT:-$REPO_ROOT}/logs}
RUN_LOG=${VAGEN_RUN_LOG:-$LOG_ROOT/$EXPERIMENT_NAME.log}
SERVER_PORT=${SERVER_PORT:-5000}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-300}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-8}
N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-4}
WANDB_MODE=${WANDB_MODE:-online}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-32}
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE=${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-4}
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.1}
ROLLOUT_MINI_BATCH_SIZE=${ROLLOUT_MINI_BATCH_SIZE:-64}
ROLLOUT_TIMEOUT=${ROLLOUT_TIMEOUT:-500}
ROLLOUT_LIMIT_MM_PER_PROMPT=${ROLLOUT_LIMIT_MM_PER_PROMPT:-20}
SAVE_FREQ=${SAVE_FREQ:-50}
TEST_FREQ=${TEST_FREQ:-20}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}
VAL_GENERATIONS_TO_LOG_TO_WANDB=${VAL_GENERATIONS_TO_LOG_TO_WANDB:-8}
REMOVE_PREVIOUS_CKPT_IN_SAVE=${REMOVE_PREVIOUS_CKPT_IN_SAVE:-False}
FORCE_GEN_DATA=${FORCE_GEN_DATA:-0}
FORCE_GEN_ARGS=()
if [ "$FORCE_GEN_DATA" = "1" ]; then
    FORCE_GEN_ARGS=(--force_gen)
fi

cd "$REPO_ROOT"
mkdir -p "$DATA_ROOT/$EXPERIMENT_NAME" "$CHECKPOINT_DIR" "$LOG_ROOT"

python -m vagen.env.create_dataset \
    --yaml_path "$ENV_CONFIG_PATH" \
    --train_path "$TRAIN_FILE" \
    --test_path "$TEST_FILE" \
    "${FORCE_GEN_ARGS[@]}"

python3 -m vagen.trainer.main_ppo \
    algorithm.adv_estimator=bi_level_gae \
    algorithm.high_level_gamma=0.95 \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=256 \
    data.max_trajectory_length=5000 \
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
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.limit_mm_per_prompt=$ROLLOUT_LIMIT_MM_PER_PROMPT \
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
    trainer.default_local_dir="$CHECKPOINT_DIR" \
    trainer.n_gpus_per_node=$N_GPUS_PER_NODE \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.remove_previous_ckpt_in_save=$REMOVE_PREVIOUS_CKPT_IN_SAVE \
    rollout_manager.max_turns=4 \
    rollout_manager.window_size=5 \
    rollout_manager.use_multi_turn_reward=True \
    rollout_manager.use_loss_mask=True \
    rollout_manager.use_gae_mask=True \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    trainer.val_generations_to_log_to_wandb=$VAL_GENERATIONS_TO_LOG_TO_WANDB \
    rollout_manager.n_trajectory=1 \
    rollout_manager.use_service=True \
    rollout_manager.timeout=$ROLLOUT_TIMEOUT \
    +rollout_manager.mini_batch_size=$ROLLOUT_MINI_BATCH_SIZE \
    rollout_manager.base_url="http://localhost:$SERVER_PORT" \
    2>&1 | tee "$RUN_LOG"
