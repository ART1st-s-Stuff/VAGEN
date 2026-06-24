#!/bin/bash
#SBATCH --job-name=qwen-bug-repro-smoke
#SBATCH --output=/project/peilab/atst/nimloth.qwen-bug-repro/outputs/experiments/training/baseline/slurm/%x_%j.out
#SBATCH --error=/project/peilab/atst/nimloth.qwen-bug-repro/outputs/experiments/training/baseline/slurm/%x_%j.err
#SBATCH --time=01:30:00
#SBATCH --account=peilab
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --requeue

# VAGEN qwen-bug-repro smoke: default Qwen + WM prompt (worldmodeling) +
# max_actions_per_step=5 + max_turns=4 + bi_level_gae, 3 navigation samples,
# 2 PPO steps, plus a vision prefix-invariance probe.
#
# Runs from a side VAGEN worktree (nimloth/qwen-bug-repro branch) so the main
# checkout (used by retry14) is untouched. verl is a side checkout of the
# legacy verl commit 2b46f25 (has compute_bi_level_gae_advantage_return).

set -euo pipefail

REPO=${REPO:-/project/peilab/atst/nimloth.qwen-bug-repro}
SCRIPTDIR=${REPO}/experiments/training/baseline
VAGEN_DIR=${VAGEN_DIR:-/project/peilab/atst/nimloth/external/VAGEN.qwen-bug-repro}
VERL_DIR=${VERL_DIR:-${VAGEN_DIR}/verl}
EXP_DIR=${VAGEN_DIR}/experiments/qwen_bug_repro
# Reuse the existing venv from the main root checkout (absolute path); the root
# worktree itself has no .venv (gitignored, lives only in the main checkout).
VENV=${VENV:-/project/peilab/atst/nimloth/.venv}

RUN_DATE=${RUN_DATE:-$(date +%Y-%m-%d)}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen_bug_repro_smoke}
RUN_DIR=${RUN_DIR:-${REPO}/outputs/experiments/training/baseline/${RUN_DATE}/${EXPERIMENT_NAME}}
DATA_DIR=${RUN_DIR}/data
SAVE_CHECKPOINT_DIR=${RUN_DIR}/checkpoints
LOG_FILE=${RUN_DIR}/${EXPERIMENT_NAME}.log

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}
PORT=${PORT:-8400}
ENV_MAX_WORKERS=${ENV_MAX_WORKERS:-8}
TOTAL_STEPS=${TOTAL_STEPS:-2}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-3}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-3}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-3}

mkdir -p "${RUN_DIR}" "${DATA_DIR}" "${SAVE_CHECKPOINT_DIR}" \
         "${REPO}/outputs/experiments/training/baseline/slurm" \
         "${RUN_DIR}/vision_probe"

# common_env.sh sets PYTHONPATH to ${REPO}/external/VAGEN/verl; we override below.
# shellcheck disable=SC1091
source "${SCRIPTDIR}/common_env.sh"
export PYTHONPATH="${REPO}/src:${VAGEN_DIR}:${VERL_DIR}:${PYTHONPATH}"
# Activate the reused venv from the main root checkout (common_env.sh expects
# caller to have activated).
# shellcheck disable=SC1091
source "${VENV}/bin/activate"

# Install together (LLM-as-judge dep, needed for vagen imports; computed node DNS works)
pip install together 2>&1 | tail -3

cd "${VAGEN_DIR}"

{
  echo "=== qwen-bug-repro smoke $(date) ==="
  echo "job=${SLURM_JOB_ID} node=$(hostname) RUN_DIR=${RUN_DIR}"
  echo "VAGEN_DIR=${VAGEN_DIR} VERL_DIR=${VERL_DIR}"
  echo "vagen_commit=$(git rev-parse HEAD)"
  echo "verl_commit=$(git -C ${VERL_DIR} rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "model=${MODEL_PATH}"
  echo "settings: prompt_format=worldmodeling max_actions_per_step=5 max_turns=4"
  echo "          adv_estimator=bi_level_gae high_level_gamma=0.95 gamma=1.0"
  echo "          train_batch=${TRAIN_BATCH_SIZE} total_steps=${TOTAL_STEPS}"
} | tee -a "${LOG_FILE}"

# --- 1. generate 3-sample dataset ---
echo "=== generating dataset ===" | tee -a "${LOG_FILE}"
python -m vagen.env.create_dataset \
  --yaml_path "${EXP_DIR}/dataset_train.yaml" \
  --force_gen \
  --seed 42 \
  --train_path "${DATA_DIR}/train.parquet" \
  --test_path "${DATA_DIR}/train_unused.parquet" 2>&1 | tee -a "${LOG_FILE}"
python -m vagen.env.create_dataset \
  --yaml_path "${EXP_DIR}/dataset_val.yaml" \
  --force_gen \
  --seed 42 \
  --train_path "${DATA_DIR}/val_unused_train.parquet" \
  --test_path "${DATA_DIR}/val.parquet" 2>&1 | tee -a "${LOG_FILE}"

# --- 2. start env server on GPU0 (AI2-THOR needs Vulkan setup) ---
ENV_LOG="${RUN_DIR}/env_server_${SLURM_JOB_ID}.log"
ENV_PID=""
cleanup() {
  set +e
  echo "=== cleanup $(date) ===" | tee -a "${LOG_FILE}"
  if [ -n "${ENV_PID}" ]; then kill "${ENV_PID}" 2>/dev/null || true; fi
  pkill -f "vagen.server.server" 2>/dev/null || true
}
trap cleanup EXIT

echo "=== starting env server on GPU0 port=${PORT} ===" | tee -a "${LOG_FILE}"
(
  # shellcheck disable=SC1091
  source "${SCRIPTDIR}/setup_ai2thor_env.sh"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 python -m vagen.server.server \
    server.port=${PORT} \
    navigation.devices='[0]' \
    navigation.max_workers=${ENV_MAX_WORKERS} \
    use_state_reward=False \
    > "${ENV_LOG}" 2>&1
) &
ENV_PID=$!

ok=0
for i in $(seq 1 120); do
  if curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    echo "env health OK tries=${i}" | tee -a "${LOG_FILE}"
    ok=1; break
  fi
  sleep 3
done
if [ "${ok}" -ne 1 ]; then
  echo "ERROR env server health failed; tail env log:" | tee -a "${LOG_FILE}"
  tail -200 "${ENV_LOG}" | tee -a "${LOG_FILE}" || true
  exit 4
fi

# --- 3. start Ray (dedicated port+temp-dir to avoid colliding with hold jobs) ---
RAY_PORT=6380
RAY_TMPDIR="/tmp/ray_smoke_${SLURM_JOB_ID}"
NODE_IP=$(hostname -I | awk '{print $1}')
echo "=== starting Ray head on ${NODE_IP}:${RAY_PORT} tmp=${RAY_TMPDIR} ===" | tee -a "${LOG_FILE}"
ray start --head \
    --port=${RAY_PORT} \
    --num-cpus=16 \
    --num-gpus=1 \
    --node-ip-address="${NODE_IP}" \
    --temp-dir="${RAY_TMPDIR}" \
    --include-dashboard=false 2>&1 | tee -a "${LOG_FILE}"
sleep 10
export RAY_ADDRESS="${NODE_IP}:${RAY_PORT}"
echo "=== Ray ready at ${RAY_ADDRESS} ===" | tee -a "${LOG_FILE}"

# --- 4. run PPO smoke on GPU1 ---
echo "=== starting PPO smoke on GPU1 ===" | tee -a "${LOG_FILE}"
set +e
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python -m vagen.trainer.main_ppo \
  algorithm.adv_estimator=bi_level_gae \
  algorithm.high_level_gamma=0.95 \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_files="${DATA_DIR}/train.parquet" \
  data.val_files="${DATA_DIR}/val.parquet" \
  data.train_batch_size=${TRAIN_BATCH_SIZE} \
  data.val_batch_size=${VAL_BATCH_SIZE} \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  data.max_trajectory_length=4096 \
  data.image_key=images \
  data.truncation=left \
  actor_rollout_ref.model.path=${MODEL_PATH} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.n=1 \
  actor_rollout_ref.rollout.top_p=0.95 \
  actor_rollout_ref.rollout.temperature=0.7 \
  actor_rollout_ref.rollout.limit_mm_per_prompt=4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.optim.lr=1e-5 \
  critic.model.path=${MODEL_PATH} \
  critic.model.enable_gradient_checkpointing=True \
  critic.ppo_micro_batch_size_per_gpu=1 \
  critic.model.fsdp_config.param_offload=True \
  critic.model.fsdp_config.optimizer_offload=True \
  reward_model.enable=False \
  trainer.critic_warmup=0 \
  trainer.logger=['console'] \
  trainer.project_name=nimloth_qwen_bug_repro \
  trainer.experiment_name=${EXPERIMENT_NAME} \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=${TOTAL_STEPS} \
  trainer.save_freq=10 \
  trainer.test_freq=1 \
  trainer.default_local_dir=${SAVE_CHECKPOINT_DIR} \
  trainer.val_before_train=True \
  trainer.val_generations_to_log_to_wandb=0 \
  rollout_manager.max_turns=4 \
  rollout_manager.window_size=4 \
  rollout_manager.n_trajectory=1 \
  rollout_manager.use_service=True \
  rollout_manager.base_url="http://localhost:${PORT}" \
  rollout_manager.timeout=900 \
  rollout_manager.max_workers=${ENV_MAX_WORKERS} \
  rollout_manager.use_loss_mask=True \
  rollout_manager.use_gae_mask=True \
  rollout_manager.use_multi_turn_reward=True \
  +rollout_manager.mini_batch_size=${VAL_BATCH_SIZE} \
  2>&1 | tee -a "${LOG_FILE}"
PPO_RC=${PIPESTATUS[0]}
set -e
echo "=== PPO smoke finished rc=${PPO_RC} $(date) ===" | tee -a "${LOG_FILE}"

# --- 4. vision prefix-invariance probe (env server still up) ---
echo "=== running vision prefix-invariance probe ===" | tee -a "${LOG_FILE}"
set +e
CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 python "${EXP_DIR}/verify_vagen_vision_prefix_invariance.py" \
  --model-path ${MODEL_PATH} \
  --base-url "http://localhost:${PORT}" \
  --max-turns 4 \
  --max-actions-per-step 5 \
  --out-dir "${RUN_DIR}/vision_probe" \
  2>&1 | tee -a "${LOG_FILE}"
PROBE_RC=${PIPESTATUS[0]}
set -e
echo "=== vision probe finished rc=${PROBE_RC} $(date) ===" | tee -a "${LOG_FILE}"

echo "=== smoke summary: ppo_rc=${PPO_RC} probe_rc=${PROBE_RC} ===" | tee -a "${LOG_FILE}"
# Non-zero PPO rc is reported but probe may still be useful; do not mask probe.
exit ${PPO_RC}
