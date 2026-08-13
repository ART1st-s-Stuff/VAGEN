"""Selected-action critic objectives for the provisional joint policy."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class SelectedActionHuberLoss:
    """Huber loss and audited tensors for executed actions only."""

    loss: Any
    per_sample_loss: Any
    selected_action_values: Any
    detached_targets: Any


def selected_action_huber_loss(
    all_action_values: Any,
    executed_action_ids: Any,
    return_targets: Any,
    *,
    delta: float,
    reduction: Literal["none", "mean", "sum"],
) -> SelectedActionHuberLoss:
    """Regress only executed-action Q against stop-gradient return targets.

    ``delta`` and ``reduction`` are deliberately mandatory because the project
    has not selected training defaults. Unexecuted action slots are never read.
    """

    import torch

    for field, value in (
        ("all_action_values", all_action_values),
        ("executed_action_ids", executed_action_ids),
        ("return_targets", return_targets),
    ):
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"joint critic {field} must be a torch Tensor")
    supported_float_dtypes = {
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    }
    if all_action_values.dtype not in supported_float_dtypes:
        raise ValueError(
            "joint critic all_action_values must use a supported real floating dtype"
        )
    if return_targets.dtype not in supported_float_dtypes:
        raise ValueError(
            "joint critic return_targets must use a supported real floating dtype"
        )
    if all_action_values.ndim != 2:
        raise ValueError("joint critic all_action_values must have shape [batch, actions]")
    if executed_action_ids.ndim != 1 or return_targets.ndim != 1:
        raise ValueError(
            "joint critic executed_action_ids and return_targets must have shape [batch]"
        )
    batch_size, action_count = all_action_values.shape
    if batch_size <= 0 or action_count <= 0:
        raise ValueError("joint critic batch and action axes must be non-empty")
    if executed_action_ids.shape[0] != batch_size or return_targets.shape[0] != batch_size:
        raise ValueError("joint critic tensor shapes must share one batch size")
    if executed_action_ids.dtype == torch.bool or executed_action_ids.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("joint critic executed_action_ids must have integer dtype")
    if not math.isfinite(float(delta)) or float(delta) <= 0.0:
        raise ValueError("joint critic Huber delta must be finite and positive")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("joint critic reduction must be none, mean, or sum")

    action_ids = executed_action_ids.to(
        device=all_action_values.device,
        dtype=torch.long,
    )
    if (action_ids < 0).any() or (action_ids >= action_count).any():
        raise ValueError("joint critic executed_action_ids are outside action space")
    selected_action_values = all_action_values.gather(
        dim=-1,
        index=action_ids.unsqueeze(-1),
    ).squeeze(-1)
    if not torch.isfinite(selected_action_values).all():
        raise ValueError("joint critic selected action values must be finite")

    detached_targets = return_targets.detach().to(
        device=all_action_values.device,
        dtype=all_action_values.dtype,
    )
    if not torch.isfinite(detached_targets).all():
        raise ValueError("joint critic return targets must be finite")

    # Compute low-precision critic losses in FP32. The clamped decomposition
    # avoids evaluating an overflowing quadratic branch for linear-region
    # samples, which ``torch.where`` would still do before selection.
    loss_dtype = (
        torch.float32
        if all_action_values.dtype in {torch.float16, torch.bfloat16}
        else all_action_values.dtype
    )
    selected_for_loss = selected_action_values.to(dtype=loss_dtype)
    targets_for_loss = detached_targets.to(dtype=loss_dtype)
    absolute_error = (selected_for_loss - targets_for_loss).abs()
    delta_tensor = absolute_error.new_tensor(float(delta))
    if not torch.isfinite(delta_tensor) or float(delta_tensor) <= 0.0:
        raise ValueError(
            "joint critic Huber delta must remain finite and positive in loss dtype"
        )
    quadratic_error = absolute_error.clamp(max=delta_tensor)
    linear_error = absolute_error - quadratic_error
    per_sample_loss = (
        0.5 * quadratic_error.square() + delta_tensor * linear_error
    )
    if not torch.isfinite(per_sample_loss).all():
        raise ValueError("joint critic Huber loss must be finite")
    if reduction == "none":
        loss = per_sample_loss
    elif reduction == "mean":
        loss = per_sample_loss.mean()
    else:
        loss = per_sample_loss.sum()

    return SelectedActionHuberLoss(
        loss=loss,
        per_sample_loss=per_sample_loss,
        selected_action_values=selected_action_values,
        detached_targets=detached_targets,
    )


__all__ = ["SelectedActionHuberLoss", "selected_action_huber_loss"]
