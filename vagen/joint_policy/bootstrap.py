"""Explicit initial critic/snapshot bootstrap shared by trainer and smoke."""

from __future__ import annotations

from typing import Any

from .contract import FrozenQGuidedPolicyConfig
from .training_contract import JointTrainingConfig


def build_initial_joint_snapshot_state(
    *,
    tokenizer: Any,
    policy_config: FrozenQGuidedPolicyConfig,
    training_config: JointTrainingConfig,
) -> dict[str, Any]:
    import torch

    from nimloth.latent import LatentActionTokens, special_token_ids
    from nimloth.training.rl.joint_critic import (
        create_frozen_critic_snapshot,
        export_frozen_critic_snapshot,
        load_joint_action_value_critic,
    )
    from vagen.envs.navigation.utils.nimloth_format import ACTION_NAMES

    if not isinstance(policy_config, FrozenQGuidedPolicyConfig):
        raise TypeError("joint bootstrap requires FrozenQGuidedPolicyConfig")
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
        contract_id=policy_config.contract_id(
            "navigation_v1",
            ACTION_NAMES,
            action_token_ids,
        ),
        score_dtype=policy_config.score_dtype,
    )
    return export_frozen_critic_snapshot(snapshot).to_mapping()


__all__ = ["build_initial_joint_snapshot_state"]
