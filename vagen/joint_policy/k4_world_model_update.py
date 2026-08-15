"""Differentiable replicated K4 world-model and selected-action critic update."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .critic_loss import selected_action_huber_loss
from .k4_training_contract import K4WorldModelTrainingConfig


@dataclass(frozen=True)
class K4WorldModelLossTerms:
    state_window_loss_sum: torch.Tensor
    dino_window_loss_sum: torch.Tensor
    window_count: torch.Tensor
    critic_loss_sum: torch.Tensor
    critic_valid_count: torch.Tensor
    sigreg_loss: torch.Tensor
    sigreg_valid_count: torch.Tensor
    all_action_values: torch.Tensor


class K4WorldModelUpdateModule(nn.Module):
    """Own the trainable projector/predictor/ValueHead and all three losses."""

    def __init__(self, model: nn.Module, config: K4WorldModelTrainingConfig) -> None:
        super().__init__()
        if not isinstance(config, K4WorldModelTrainingConfig):
            raise TypeError("K4 WM update requires K4WorldModelTrainingConfig")
        from nimloth.training.rl.joint_planner import JointWorldModelCritic
        from nimloth.wm import SequenceSIGReg

        if not isinstance(model, JointWorldModelCritic):
            raise TypeError("K4 WM update requires JointWorldModelCritic")
        if model.wm_predictor.config.history_size != 1:
            raise ValueError("K4 online WM update requires history_size=1")
        if model.wm_predictor.config.emb_dim != config.dino_identity["hidden_size"]:
            raise ValueError("K4 WM state dim must match DINO teacher hidden size")
        if model.wm_predictor.config.grid_tokens != config.dino_identity["grid_size"] ** 2:
            raise ValueError("K4 WM grid must match DINO teacher grid size")
        self.model = model
        self.config = config
        self.sigreg = SequenceSIGReg(
            knots=config.sigreg_knots,
            num_proj=config.sigreg_num_proj,
        )

    def forward(
        self,
        *,
        current_hidden: torch.Tensor,
        future_hidden: torch.Tensor,
        future_action_ids: torch.Tensor,
        future_valid_mask: torch.Tensor,
        valid_row_mask: torch.Tensor,
        guided_action_ids: torch.Tensor,
        critic_returns: torch.Tensor,
        future_dino_grid_targets: torch.Tensor,
        sigreg_seed: int,
    ) -> K4WorldModelLossTerms:
        if current_hidden.ndim != 3:
            raise ValueError("K4 WM current hidden must have shape (B,N,H)")
        batch_size, grid_tokens, hidden_dim = current_hidden.shape
        horizon = self.config.prediction_horizon
        if future_hidden.shape != (batch_size, horizon, grid_tokens, hidden_dim):
            raise ValueError("K4 WM future hidden shape mismatch")
        if future_action_ids.shape != (batch_size, horizon):
            raise ValueError("K4 WM future actions must have shape (B,4)")
        if future_valid_mask.shape != (batch_size, horizon):
            raise ValueError("K4 WM future valid mask must have shape (B,4)")
        for field, value in (
            ("valid_row_mask", valid_row_mask),
            ("guided_action_ids", guided_action_ids),
            ("critic_returns", critic_returns),
        ):
            if value.shape != (batch_size,):
                raise ValueError(f"K4 WM {field} must have shape (B,)")
        state_dim = self.model.wm_predictor.config.emb_dim
        if future_dino_grid_targets.shape != (
            batch_size,
            horizon,
            grid_tokens,
            state_dim,
        ):
            raise ValueError("K4 WM DINO-grid target shape mismatch")
        if not isinstance(sigreg_seed, int) or isinstance(sigreg_seed, bool) or sigreg_seed < 0:
            raise ValueError("K4 WM sigreg_seed must be a non-negative int")
        if not torch.isfinite(current_hidden).all() or not torch.isfinite(future_hidden).all():
            raise ValueError("K4 WM hidden tensors must be finite")
        if not torch.isfinite(future_dino_grid_targets).all():
            raise ValueError("K4 WM DINO-grid targets must be finite")
        row_mask = valid_row_mask.to(device=current_hidden.device, dtype=torch.bool)
        future_mask = future_valid_mask.to(device=current_hidden.device, dtype=torch.bool)
        if torch.any(future_mask[:, 1:] & ~future_mask[:, :-1]):
            raise ValueError("K4 WM valid future depth must be a contiguous prefix")
        current_state = self.model.project_state(current_hidden)
        with torch.no_grad():
            target_state = self.model.project_state_sequence(future_hidden).detach()
        state_loss_sum = current_state.sum() * 0.0
        dino_loss_sum = current_state.sum() * 0.0
        window_count = torch.zeros((), device=current_state.device, dtype=torch.long)
        empty_previous = future_action_ids.new_empty((batch_size, 0))
        for depth in range(1, horizon + 1):
            selected_rows = row_mask & future_mask[:, depth - 1]
            count = selected_rows.sum()
            if int(count.item()) == 0:
                continue
            predictions = self.model.wm_predictor.rollout_from_history(
                current_state[selected_rows].unsqueeze(1),
                empty_previous[selected_rows],
                future_action_ids[selected_rows, :depth],
            )
            expected = target_state[selected_rows, :depth]
            dino = future_dino_grid_targets[selected_rows, :depth].detach().to(
                device=predictions.device,
                dtype=torch.float32,
            )
            state_per_window = (predictions.float() - expected.float()).square().flatten(1).mean(-1)
            dino_per_window = (predictions.float() - dino).square().flatten(1).mean(-1)
            state_loss_sum = state_loss_sum + state_per_window.sum()
            dino_loss_sum = dino_loss_sum + dino_per_window.sum()
            window_count = window_count + count
        if int(window_count.item()) < 1:
            state_loss_sum = state_loss_sum + _zero_parameter_anchor(
                self.model.wm_predictor,
                current_state,
            )
        _require_global_positive_count(
            window_count,
            "K4 WM update contains no valid 1--4-step window",
        )

        all_action_values = self.model.predict_action_values(current_state)
        critic = selected_action_huber_loss(
            all_action_values,
            guided_action_ids,
            critic_returns,
            delta=self.config.selected_action_huber_delta,
            reduction="none",
        )
        critic_loss_sum = (
            critic.per_sample_loss
            * row_mask.to(dtype=critic.per_sample_loss.dtype)
        ).sum()
        critic_valid_count = row_mask.sum()
        _require_global_positive_count(
            critic_valid_count,
            "K4 critic update contains no valid executed turn",
        )

        one_step = row_mask & future_mask[:, 0]
        next_online = self.model.project_state(future_hidden[one_step, 0])
        current_sigreg = self.model.sigreg_state(current_state[one_step].detach())
        next_sigreg = self.model.sigreg_state(next_online)
        from nimloth.training.sft2.algorithm import (
            gather_global_sigreg_states,
            shared_sigreg_rng,
        )

        global_current, global_next, global_count = gather_global_sigreg_states(
            current_sigreg,
            next_sigreg,
            torch.ones(next_sigreg.shape[0], device=next_sigreg.device, dtype=torch.bool),
        )
        with shared_sigreg_rng(sigreg_seed, global_next.device):
            raw_sigreg = self.sigreg(
                torch.stack((global_current, global_next), dim=1)
            )
        sigreg_loss = (
            global_next.sum() * 0.0
            if raw_sigreg is None
            else raw_sigreg
        )
        values = (
            state_loss_sum,
            dino_loss_sum,
            critic_loss_sum,
            sigreg_loss,
            all_action_values,
        )
        if any(not torch.isfinite(value).all() for value in values):
            raise RuntimeError("K4 world-model loss or value became non-finite")
        return K4WorldModelLossTerms(
            state_window_loss_sum=state_loss_sum,
            dino_window_loss_sum=dino_loss_sum,
            window_count=window_count,
            critic_loss_sum=critic_loss_sum,
            critic_valid_count=critic_valid_count,
            sigreg_loss=sigreg_loss,
            sigreg_valid_count=torch.tensor(
                global_count,
                device=current_state.device,
                dtype=torch.long,
            ),
            all_action_values=all_action_values,
        )


def _require_global_positive_count(
    local_count: torch.Tensor,
    message: str,
) -> torch.Tensor:
    if local_count.ndim != 0 or local_count.dtype != torch.long:
        raise ValueError("K4 global count input must be a scalar long tensor")
    global_count = local_count.detach().clone()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM)
    if int(global_count.item()) < 1:
        raise ValueError(message)
    return global_count


def _zero_parameter_anchor(module: nn.Module, reference: torch.Tensor) -> torch.Tensor:
    anchor = reference.sum() * 0.0
    for parameter in module.parameters():
        if parameter.numel() < 1:
            raise ValueError("K4 predictor contains an empty parameter")
        anchor = anchor + parameter.reshape(-1)[0].to(
            device=reference.device,
            dtype=reference.dtype,
        ) * 0.0
    return anchor


def build_k4_planning_optimizer(
    module: K4WorldModelUpdateModule,
) -> torch.optim.Optimizer:
    """Build exactly one AdamW with three named parameter groups."""

    if not isinstance(module, K4WorldModelUpdateModule):
        raise TypeError("K4 planning optimizer requires K4WorldModelUpdateModule")
    config = module.config.optimizer
    groups = []
    for name, child, lr in (
        ("state_projector", module.model.state_proj, config.projector_lr),
        ("wm_predictor", module.model.wm_predictor, config.predictor_lr),
        ("value_head", module.model.value_head, config.value_head_lr),
    ):
        parameters = [parameter for parameter in child.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError(f"K4 planning optimizer group {name} is empty")
        groups.append({"name": name, "params": parameters, "lr": lr})
    parameter_ids = [id(parameter) for group in groups for parameter in group["params"]]
    if len(parameter_ids) != len(set(parameter_ids)):
        raise ValueError("K4 planning optimizer parameter groups overlap")
    return torch.optim.AdamW(
        groups,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


__all__ = [
    "K4WorldModelLossTerms",
    "K4WorldModelUpdateModule",
    "build_k4_planning_optimizer",
]
