#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODES="${MODES:-world_state_with_decoder,direct_latent_mlp,lewm_latent_dynamics}"
REQUIRE_LEWM="${REQUIRE_LEWM:-false}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LATENT_DIM="${LATENT_DIM:-16}"
WORLD_STATE_DIM="${WORLD_STATE_DIM:-12}"
HIDDEN_DIM="${HIDDEN_DIM:-24}"
NUM_ACTIONS="${NUM_ACTIONS:-8}"
HORIZON="${HORIZON:-3}"

cd "$REPO_ROOT"
"$PYTHON_BIN" - <<'PY'
import importlib.util
import os
import pathlib
from types import SimpleNamespace

import torch

repo_root = pathlib.Path.cwd()
wm_path = repo_root / "verl/verl/workers/roles/utils/world_model.py"
spec = importlib.util.spec_from_file_location("wm", wm_path)
wm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wm)

modes = [m.strip() for m in os.getenv("MODES", "world_state_with_decoder,direct_latent_mlp,lewm_latent_dynamics").split(",") if m.strip()]
require_lewm = os.getenv("REQUIRE_LEWM", "false").lower() == "true"

bs = int(os.getenv("BATCH_SIZE", "2"))
latent_dim = int(os.getenv("LATENT_DIM", "16"))
world_state_dim = int(os.getenv("WORLD_STATE_DIM", "12"))
hidden_dim = int(os.getenv("HIDDEN_DIM", "24"))
num_actions = int(os.getenv("NUM_ACTIONS", "8"))
horizon = int(os.getenv("HORIZON", "3"))

print(f"[phase2-smoke] modes={modes}")
for mode in modes:
    try:
        predictor = wm.build_latent_predictor(
            predictor_mode=mode,
            latent_dim=latent_dim,
            num_actions=num_actions,
            world_state_dim=world_state_dim,
            hidden_dim=hidden_dim,
            multi_step_horizon=horizon,
        )
    except ModuleNotFoundError as e:
        if mode == "lewm_latent_dynamics" and not require_lewm:
            print(f"[phase2-smoke] skip mode={mode}: missing optional dependency ({e})")
            continue
        raise

    latent = torch.randn(bs, latent_dim, requires_grad=True)
    action_labels = torch.randint(0, num_actions, (bs,), dtype=torch.long)
    step_rewards = torch.randn(bs)
    next_latent = torch.randn(bs, latent_dim)
    cfg = SimpleNamespace(reward_loss_coef=1.0, state_loss_coef=1.0)

    aux_loss, metrics, handled = wm.compute_world_model_aux_loss(
        predictor=predictor,
        latent=latent,
        action_labels=action_labels,
        step_rewards=step_rewards,
        config=cfg,
        next_latent=next_latent,
    )
    assert handled and aux_loss is not None, f"mode={mode}: world model branch not handled"
    aux_loss.backward()

    action_seq = torch.randint(0, num_actions, (bs, horizon), dtype=torch.long)
    rollout = predictor.forward_multi_step(latent.detach(), action_seq=action_seq, horizon=horizon)
    assert "pred_next_latent_seq" in rollout, f"mode={mode}: missing pred_next_latent_seq"
    expected_shape = (bs, horizon, latent_dim)
    assert tuple(rollout["pred_next_latent_seq"].shape) == expected_shape, (
        f"mode={mode}: shape mismatch, got={tuple(rollout['pred_next_latent_seq'].shape)}, expected={expected_shape}"
    )
    print(f"[phase2-smoke] mode={mode} ok, metrics={sorted(metrics.keys())}")

print("[phase2-smoke] all selected modes passed.")
PY

