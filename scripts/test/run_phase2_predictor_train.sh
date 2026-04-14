#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Supported modes:
# - world_state_with_decoder
# - direct_latent_mlp
# - lewm_latent_dynamics
# - world_state_mlp (legacy baseline-compatible predictor path)
PREDICTOR_MODE="${PREDICTOR_MODE:-world_state_with_decoder}"
PREDICTOR_HORIZON="${PREDICTOR_HORIZON:-2}"

# Forward common knobs to predictor_debug.sh (can be overridden by env).
export PREDICTOR_MODE
export PREDICTOR_HORIZON
export NUM_ACTIONS="${NUM_ACTIONS:-8}"
export WORLD_STATE_DIM="${WORLD_STATE_DIM:-256}"
export TRAIN_STEPS="${TRAIN_STEPS:-1}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-2}"
export PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
export NUM_GPUS="${NUM_GPUS:-2}"
export NUM_NODES="${NUM_NODES:-1}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.20}"

cd "$REPO_ROOT"
echo "[phase2-train] mode=$PREDICTOR_MODE horizon=$PREDICTOR_HORIZON"
echo "[phase2-train] train_steps=$TRAIN_STEPS num_gpus=$NUM_GPUS"
exec "$REPO_ROOT/predictor_debug.sh"

