"""Torch implementation of the provisional frozen-Q guided distribution."""

from __future__ import annotations

from typing import Any

from .contract import FrozenQGuidedPolicyConfig


def frozen_q_guided_log_probs(
    prior_logits: Any,
    frozen_all_action_q: Any,
    config: FrozenQGuidedPolicyConfig,
) -> dict[str, Any]:
    """Return prior and guided log-probs while enforcing Scheme B gradients."""

    import torch

    prior_dtype = str(prior_logits.dtype).removeprefix("torch.")
    if prior_dtype != config.score_dtype:
        raise ValueError(
            "joint policy prior dtype does not match score_dtype contract: "
            f"tensor={prior_dtype}, contract={config.score_dtype}"
        )
    if prior_logits.ndim < 1 or prior_logits.shape[-1] <= 0:
        raise ValueError("joint policy prior_logits must end with a non-empty action axis")
    if prior_logits.shape != frozen_all_action_q.shape:
        raise ValueError(
            "joint policy prior logits and frozen Q must have identical shapes: "
            f"prior={tuple(prior_logits.shape)}, q={tuple(frozen_all_action_q.shape)}"
        )
    if not torch.isfinite(prior_logits).all():
        raise ValueError("joint policy prior_logits must be finite")
    if not torch.isfinite(frozen_all_action_q).all():
        raise ValueError("joint policy frozen Q must be finite")

    policy_prior_logits = prior_logits
    scaled_prior_logits = policy_prior_logits / config.prior_temperature
    if not torch.isfinite(scaled_prior_logits).all():
        raise ValueError("joint policy scaled prior logits must be finite")
    q_guidance = frozen_all_action_q.detach().to(
        device=scaled_prior_logits.device,
        dtype=scaled_prior_logits.dtype,
    )
    if not torch.isfinite(q_guidance).all():
        raise ValueError("joint policy cast frozen Q must be finite")
    guided_logits = config.alpha * scaled_prior_logits + config.beta * q_guidance
    if not torch.isfinite(guided_logits).all():
        raise ValueError("joint policy guided logits must be finite")
    prior_log_probs = torch.log_softmax(scaled_prior_logits, dim=-1)
    guided_log_probs = torch.log_softmax(guided_logits, dim=-1)
    if not torch.isfinite(prior_log_probs).all() or not torch.isfinite(
        guided_log_probs
    ).all():
        raise ValueError("joint policy log-probs must be finite")
    return {
        "prior_log_probs": prior_log_probs,
        "guided_logits": guided_logits,
        "guided_log_probs": guided_log_probs,
    }


__all__ = ["frozen_q_guided_log_probs"]
