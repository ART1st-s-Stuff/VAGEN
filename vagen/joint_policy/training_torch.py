"""Torch losses for actual guided actions and reference-token regularization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contract import FrozenQGuidedPolicyConfig
from .torch_policy import frozen_q_guided_log_probs


@dataclass(frozen=True)
class GuidedActionPPOTerms:
    policy_loss_sum: Any
    entropy_sum: Any
    valid_count: Any
    selected_current_log_probs: Any
    ratios: Any
    clipped_ratios: Any


@dataclass(frozen=True)
class TokenKLTerms:
    kl_sum: Any
    valid_token_count: Any


def guided_action_ppo_terms(
    *,
    current_prior_logits: Any,
    frozen_all_action_q: Any,
    guided_action_ids: Any,
    behavior_guided_log_probs: Any,
    advantages: Any,
    valid_mask: Any,
    policy_config: FrozenQGuidedPolicyConfig,
    clip_ratio: float,
) -> GuidedActionPPOTerms:
    """Return summed clipped PPO and entropy terms for executed guided actions."""

    import torch

    if not isinstance(policy_config, FrozenQGuidedPolicyConfig):
        raise TypeError("guided PPO requires FrozenQGuidedPolicyConfig")
    if not isinstance(clip_ratio, (float, int)) or isinstance(clip_ratio, bool):
        raise ValueError("guided PPO clip_ratio must be a finite real")
    clip = float(clip_ratio)
    if not torch.isfinite(torch.tensor(clip)) or clip <= 0.0 or clip >= 1.0:
        raise ValueError("guided PPO clip_ratio must be in (0, 1)")
    if current_prior_logits.ndim != 2:
        raise ValueError("guided PPO current prior logits must have shape (B, A)")
    batch_size, action_count = current_prior_logits.shape
    if action_count < 1 or frozen_all_action_q.shape != current_prior_logits.shape:
        raise ValueError("guided PPO frozen Q must align with current prior logits")
    for field, tensor in (
        ("guided_action_ids", guided_action_ids),
        ("behavior_guided_log_probs", behavior_guided_log_probs),
        ("advantages", advantages),
        ("valid_mask", valid_mask),
    ):
        if tuple(tensor.shape) != (batch_size,):
            raise ValueError(f"guided PPO {field} must have shape (B,)")
    if guided_action_ids.dtype != torch.long:
        raise ValueError("guided PPO action IDs must be torch.long")
    if torch.any(guided_action_ids < 0) or torch.any(guided_action_ids >= action_count):
        raise ValueError("guided PPO action ID is outside the action table")
    if not torch.isfinite(behavior_guided_log_probs).all():
        raise ValueError("guided PPO behavior log-probs must be finite")
    if not torch.isfinite(advantages).all():
        raise ValueError("guided PPO advantages must be finite")
    mask = valid_mask.to(device=current_prior_logits.device, dtype=torch.bool)
    valid_count = mask.sum()

    outputs = frozen_q_guided_log_probs(
        current_prior_logits,
        frozen_all_action_q,
        policy_config,
    )
    guided_log_probs = outputs["guided_log_probs"]
    selected = guided_log_probs.gather(-1, guided_action_ids.unsqueeze(-1)).squeeze(-1)
    behavior = behavior_guided_log_probs.detach().to(
        device=selected.device,
        dtype=selected.dtype,
    )
    advantage = advantages.detach().to(device=selected.device, dtype=selected.dtype)
    ratios = torch.exp(selected - behavior)
    clipped_ratios = torch.clamp(ratios, 1.0 - clip, 1.0 + clip)
    surrogate = torch.minimum(ratios * advantage, clipped_ratios * advantage)
    policy_loss_sum = -(surrogate * mask).sum()
    probabilities = guided_log_probs.exp()
    entropy = -(probabilities * guided_log_probs).sum(dim=-1)
    entropy_sum = (entropy * mask).sum()
    if not torch.isfinite(policy_loss_sum) or not torch.isfinite(entropy_sum):
        raise ValueError("guided PPO loss terms must be finite")
    return GuidedActionPPOTerms(
        policy_loss_sum=policy_loss_sum,
        entropy_sum=entropy_sum,
        valid_count=valid_count,
        selected_current_log_probs=selected,
        ratios=ratios,
        clipped_ratios=clipped_ratios,
    )


def low_variance_token_kl_terms(
    *,
    current_token_log_probs: Any,
    reference_token_log_probs: Any,
    response_mask: Any,
    valid_row_mask: Any,
) -> TokenKLTerms:
    """Sampled non-negative k3 estimate of KL(current || frozen reference)."""

    import torch

    if current_token_log_probs.ndim != 2:
        raise ValueError("token KL log-probs must have shape (B, R)")
    if reference_token_log_probs.shape != current_token_log_probs.shape:
        raise ValueError("token KL current and reference log-probs must align")
    if response_mask.shape != current_token_log_probs.shape:
        raise ValueError("token KL response mask must align with log-probs")
    if tuple(valid_row_mask.shape) != (current_token_log_probs.shape[0],):
        raise ValueError("token KL valid row mask must have shape (B,)")
    if not torch.isfinite(current_token_log_probs).all() or not torch.isfinite(
        reference_token_log_probs
    ).all():
        raise ValueError("token KL log-probs must be finite")
    mask = response_mask.to(dtype=torch.bool) & valid_row_mask.to(
        device=response_mask.device,
        dtype=torch.bool,
    ).unsqueeze(-1)
    count = mask.sum()
    log_ratio = (
        reference_token_log_probs.detach().to(
            device=current_token_log_probs.device,
            dtype=current_token_log_probs.dtype,
        )
        - current_token_log_probs
    )
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    kl_sum = (kl * mask).sum()
    if not torch.isfinite(kl_sum):
        raise ValueError("token KL must be finite")
    return TokenKLTerms(kl_sum=kl_sum, valid_token_count=count)


__all__ = [
    "GuidedActionPPOTerms",
    "TokenKLTerms",
    "guided_action_ppo_terms",
    "low_variance_token_kl_terms",
]
