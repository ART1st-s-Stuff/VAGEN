"""Explicit initial critic/snapshot bootstrap shared by trainer and smoke."""

from __future__ import annotations

from typing import Any

from .contract import FrozenQGuidedPolicyConfig
from .k4_training_contract import K4WorldModelTrainingConfig
from .planning_contract import K4MCTSGuidedPolicyConfig
from .training_contract import JointTrainingConfig


def build_initial_joint_snapshot_state(
    *,
    tokenizer: Any,
    policy_config: FrozenQGuidedPolicyConfig | K4MCTSGuidedPolicyConfig,
    training_config: JointTrainingConfig,
    k4_world_model_config: K4WorldModelTrainingConfig | None = None,
) -> dict[str, Any]:
    import torch

    from nimloth.latent import LatentActionTokens, special_token_ids
    from nimloth.training.rl.joint_critic import (
        create_frozen_critic_snapshot,
        export_frozen_critic_snapshot,
        load_joint_action_value_critic,
    )
    from vagen.envs.navigation.utils.nimloth_format import ACTION_NAMES

    if not isinstance(
        policy_config,
        (FrozenQGuidedPolicyConfig, K4MCTSGuidedPolicyConfig),
    ):
        raise TypeError("joint bootstrap requires a supported guided policy")
    if not isinstance(training_config, JointTrainingConfig):
        raise TypeError("joint bootstrap requires JointTrainingConfig")
    token_table = special_token_ids(
        tokenizer,
        latent_token_count=training_config.critic_grid_tokens,
    )
    action_token_ids = tuple(
        token_table[token] for token in LatentActionTokens().action_tokens
    )
    if len(action_token_ids) != training_config.critic_action_count:
        raise ValueError("joint bootstrap action table mismatches critic action count")
    contract_id = policy_config.contract_id(
        "navigation_v1",
        ACTION_NAMES,
        action_token_ids,
    )
    if isinstance(policy_config, K4MCTSGuidedPolicyConfig):
        if not isinstance(k4_world_model_config, K4WorldModelTrainingConfig):
            raise ValueError("K4 bootstrap requires world-model training config")
        from pathlib import Path
        from nimloth.training.rl.joint_planner import (
            FrozenMCTSPlanningConfig,
            create_frozen_planning_snapshot,
            load_frozen_planning_snapshot_file,
            load_joint_world_model_critic,
            save_frozen_planning_snapshot_file,
        )
        from vagen.joint_policy.planning_owner import (
            FROZEN_K4_PLANNER_TRANSPORT_SCHEMA,
        )

        model = load_joint_world_model_critic(
            checkpoint_root=Path(k4_world_model_config.planning_checkpoint),
            expected_qwen_hidden_dim=training_config.critic_qwen_hidden_dim,
            expected_grid_tokens=training_config.critic_grid_tokens,
            expected_state_dim=training_config.critic_state_dim,
            expected_action_count=training_config.critic_action_count,
            expected_prediction_horizon=(
                k4_world_model_config.prediction_horizon
            ),
            device=torch.device("cpu"),
            trainable=False,
        )
        snapshot = create_frozen_planning_snapshot(
            model,
            source_step=training_config.initial_snapshot_source_step,
            contract_id=contract_id,
            score_dtype=policy_config.score_dtype,
            planning_config=FrozenMCTSPlanningConfig(
                horizon=policy_config.planning_horizon,
                num_simulations=policy_config.mcts_num_simulations,
                exploration_constant=policy_config.mcts_exploration_constant,
            ),
        )
        target = (
            Path(k4_world_model_config.snapshot_transport_root)
            / f"source_step_{training_config.initial_snapshot_source_step}"
            / "frozen_k4_planner.pt"
        ).resolve()
        if target.exists():
            restored = load_frozen_planning_snapshot_file(
                target,
                device=torch.device("cpu"),
            )
            if restored.snapshot_id != snapshot.snapshot_id:
                raise ValueError("existing initial K4 transport has wrong state")
        else:
            save_frozen_planning_snapshot_file(snapshot, target)
        return {
            "schema": FROZEN_K4_PLANNER_TRANSPORT_SCHEMA,
            "transport_path": str(target),
            "snapshot_id": snapshot.snapshot_id,
            "snapshot_source_step": snapshot.source_step,
            "contract_id": snapshot.contract_id,
            "score_dtype": snapshot.score_dtype,
            "planning_horizon": policy_config.planning_horizon,
            "mcts_num_simulations": policy_config.mcts_num_simulations,
            "mcts_exploration_constant": (
                policy_config.mcts_exploration_constant
            ),
        }
    if k4_world_model_config is not None:
        raise ValueError("legacy bootstrap cannot receive K4 world-model config")
    critic = load_joint_action_value_critic(
        checkpoint_root=training_config.critic_checkpoint,
        expected_qwen_hidden_dim=training_config.critic_qwen_hidden_dim,
        expected_grid_tokens=training_config.critic_grid_tokens,
        expected_state_dim=training_config.critic_state_dim,
        expected_action_count=training_config.critic_action_count,
        device=torch.device("cpu"),
        trainable=False,
    )
    snapshot = create_frozen_critic_snapshot(
        critic,
        source_step=training_config.initial_snapshot_source_step,
        contract_id=contract_id,
        score_dtype=policy_config.score_dtype,
    )
    return export_frozen_critic_snapshot(snapshot).to_mapping()


__all__ = ["build_initial_joint_snapshot_state"]
