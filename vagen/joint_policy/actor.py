"""FSDP actor extension with replicated GPU action-value critic training."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from verl.utils.device import get_device_id
from verl.workers.actor.dp_actor import DataParallelPPOActor

from .contract import FrozenQGuidedPolicyConfig
from .critic_loss import selected_action_huber_loss
from .training_contract import JointTrainingConfig, parse_joint_training_section
from .training_torch import guided_action_ppo_terms, low_variance_token_kl_terms

JOINT_ACTOR_CHECKPOINT_SCHEMA = "vagen_joint_actor_critic_checkpoint_v1"


class JointDataParallelPPOActor(DataParallelPPOActor):
    """Guided-action PPO actor plus a DP-replicated current critic."""

    def __init__(self, config, actor_module, actor_optimizer=None):
        super().__init__(config, actor_module, actor_optimizer)
        raw = config.custom_config
        if not isinstance(raw, Mapping) or set(raw) != {"joint_policy", "joint_training"}:
            raise ValueError(
                "joint custom actor requires exact joint_policy and joint_training configs"
            )
        self.joint_training = parse_joint_training_section(raw["joint_training"])
        if self.joint_training is None:
            raise ValueError("joint custom actor requires joint_training.enabled=true")
        self.joint_policy = FrozenQGuidedPolicyConfig.from_mapping(
            raw["joint_policy"]
        )
        if self.use_fused_kernels:
            raise ValueError("joint guided actor requires use_fused_kernels=false")
        if self.config.ppo_epochs != 1:
            raise ValueError("joint guided actor requires exactly one PPO epoch")
        if self.config.use_dynamic_bsz:
            raise ValueError(
                "joint guided actor dynamic batching is not implemented; fail closed"
            )
        if actor_optimizer is None:
            raise ValueError("joint guided actor requires an actor optimizer")
        expected_actor_optim = self.joint_training.actor_optimizer
        actual_actor_optim = self.config.optim
        override = actual_actor_optim.override_optimizer_config
        if (
            actual_actor_optim.optimizer != "AdamW"
            or actual_actor_optim.optimizer_impl != "torch.optim"
            or float(actual_actor_optim.lr) != expected_actor_optim.lr
            or tuple(actual_actor_optim.betas) != expected_actor_optim.betas
            or float(actual_actor_optim.weight_decay)
            != expected_actor_optim.weight_decay
            or float(self.config.grad_clip) != expected_actor_optim.grad_clip
            or actual_actor_optim.lr_scheduler_type
            != expected_actor_optim.lr_scheduler_type
            or int(actual_actor_optim.lr_warmup_steps)
            != expected_actor_optim.lr_warmup_steps
            or float(actual_actor_optim.lr_warmup_steps_ratio)
            != expected_actor_optim.lr_warmup_steps_ratio
            or actual_actor_optim.min_lr_ratio
            != expected_actor_optim.min_lr_ratio
            or float(actual_actor_optim.num_cycles) != expected_actor_optim.num_cycles
            or not isinstance(override, Mapping)
            or set(override) != {"eps"}
            or float(override["eps"]) != expected_actor_optim.eps
        ):
            raise ValueError(
                "joint actor optimizer runtime does not match explicit training contract"
            )
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("joint replicated critic requires initialized torch.distributed")

        self._joint_world_size = dist.get_world_size()
        self._joint_rank = dist.get_rank()
        device = get_device_id()
        from nimloth.training.rl.joint_critic import load_joint_action_value_critic

        critic = load_joint_action_value_critic(
            checkpoint_root=Path(self.joint_training.critic_checkpoint),
            expected_qwen_hidden_dim=self.joint_training.critic_qwen_hidden_dim,
            expected_grid_tokens=self.joint_training.critic_grid_tokens,
            expected_state_dim=self.joint_training.critic_state_dim,
            expected_action_count=self.joint_training.critic_action_count,
            device=device,
            trainable=True,
        )
        critic.train(True)
        self.current_joint_critic = DistributedDataParallel(
            critic,
            device_ids=[device],
            output_device=device,
            broadcast_buffers=False,
        )
        optim = self.joint_training.critic_optimizer
        self.joint_critic_optimizer = torch.optim.AdamW(
            self.current_joint_critic.parameters(),
            lr=optim.lr,
            betas=optim.betas,
            eps=optim.eps,
            weight_decay=optim.weight_decay,
        )
        self._joint_completed_updates = 0
        self._joint_contract_id: str | None = None

    @property
    def joint_source_step(self) -> int:
        return (
            self.joint_training.initial_snapshot_source_step
            + self._joint_completed_updates
        )

    def update_policy(self, data):
        """Run one guided actor epoch and one selected-action critic update."""

        required = {
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "joint_action_token_ids",
            "joint_prior_token_ids",
            "joint_prior_response_indices",
            "joint_guided_action_ids",
            "joint_behavior_guided_log_probs",
            "joint_frozen_all_action_q",
            "joint_advantages",
            "joint_valid_mask",
            "joint_critic_hidden",
            "joint_critic_returns",
        }
        if self.joint_training.token_kl_coefficient > 0.0:
            required.add("joint_reference_token_log_probs")
        missing = required - set(data.batch)
        if missing:
            raise ValueError(f"joint actor batch is missing tensors: {sorted(missing)}")
        if self.config.use_kl_loss or self.config.entropy_coeff != 0:
            raise ValueError(
                "stock token PPO KL/entropy must be disabled for joint guided actor"
            )

        data = data.select(batch_keys=sorted(required))
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        metrics: dict[str, list[float]] = {}
        for mini_batch in mini_batches:
            self._update_joint_mini_batch(mini_batch, metrics)
        self.actor_optimizer.zero_grad()
        self.joint_critic_optimizer.zero_grad()
        if any(
            not torch.isfinite(parameter).all()
            for parameter in self.current_joint_critic.parameters()
        ):
            raise RuntimeError("joint critic parameters became non-finite")
        self._joint_completed_updates += 1
        metrics.setdefault("joint/completed_updates", []).append(
            float(self._joint_completed_updates)
        )
        metrics.setdefault("joint/source_step", []).append(float(self.joint_source_step))
        return metrics

    def _update_joint_mini_batch(self, mini_batch, metrics: dict[str, list[float]]) -> None:
        micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
        local_valid = mini_batch.batch["joint_valid_mask"].to(dtype=torch.long).sum().to(get_device_id())
        global_valid = local_valid.clone()
        dist.all_reduce(global_valid, op=dist.ReduceOp.SUM)
        if int(global_valid.item()) < 1:
            raise ValueError("joint PPO mini-batch contains no valid executed turns")
        if self.joint_training.token_kl_coefficient > 0.0:
            local_token_count = (
                mini_batch.batch["response_mask"].to(dtype=torch.bool)
                & mini_batch.batch["joint_valid_mask"].to(dtype=torch.bool).unsqueeze(-1)
            ).sum().to(get_device_id())
            global_token_count = local_token_count.clone()
            dist.all_reduce(global_token_count, op=dist.ReduceOp.SUM)
            if int(global_token_count.item()) < 1:
                raise ValueError("joint PPO mini-batch contains no reference KL tokens")
        else:
            global_token_count = None

        actor_scale = self._joint_world_size / float(global_valid.item())
        token_scale = (
            0.0
            if global_token_count is None
            else self._joint_world_size / float(global_token_count.item())
        )
        self.actor_optimizer.zero_grad()
        self.joint_critic_optimizer.zero_grad()
        actor_loss_value = 0.0
        policy_sum_value = 0.0
        entropy_sum_value = 0.0
        token_kl_sum_value = 0.0
        critic_sum_value = 0.0

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            tensors = micro_batch.batch
            action_tables = tensors["joint_action_token_ids"].to(dtype=torch.long)
            if action_tables.ndim != 2 or not torch.equal(
                action_tables,
                action_tables[0].expand_as(action_tables),
            ):
                raise ValueError("joint actor micro-batch must share one action token table")
            prior_indices = tensors["joint_prior_response_indices"].to(dtype=torch.long)
            prior_tokens = tensors["joint_prior_token_ids"].to(dtype=torch.long)
            actual_prior_tokens = tensors["responses"].gather(
                -1,
                prior_indices.unsqueeze(-1),
            ).squeeze(-1)
            if not torch.equal(actual_prior_tokens, prior_tokens):
                raise ValueError(
                    "joint actor prior response index does not identify the rollout prior token"
                )
            forward = self._forward_micro_batch(
                {**tensors, **micro_batch.non_tensor_batch},
                temperature=micro_batch.meta_info["temperature"],
                calculate_entropy=False,
                action_token_ids=action_tables[0],
                action_response_indices=prior_indices,
            )
            _entropy, current_token_log_probs, raw_action_logits = forward
            score_dtype = _score_dtype(self.joint_policy.score_dtype)
            action_logits = raw_action_logits.to(dtype=score_dtype)
            frozen_q = tensors["joint_frozen_all_action_q"].to(
                device=action_logits.device,
                dtype=score_dtype,
            )
            valid_mask = tensors["joint_valid_mask"].to(dtype=torch.bool)
            ppo = guided_action_ppo_terms(
                current_prior_logits=action_logits,
                frozen_all_action_q=frozen_q,
                guided_action_ids=tensors["joint_guided_action_ids"].to(dtype=torch.long),
                behavior_guided_log_probs=tensors[
                    "joint_behavior_guided_log_probs"
                ].to(dtype=score_dtype),
                advantages=tensors["joint_advantages"].to(dtype=score_dtype),
                valid_mask=valid_mask,
                policy_config=self.joint_policy,
                clip_ratio=self.joint_training.ppo_clip_ratio,
            )
            actor_loss = actor_scale * (
                ppo.policy_loss_sum
                - self.joint_training.guided_entropy_coefficient * ppo.entropy_sum
            )
            if self.joint_training.token_kl_coefficient > 0.0:
                token_kl = low_variance_token_kl_terms(
                    current_token_log_probs=current_token_log_probs,
                    reference_token_log_probs=tensors[
                        "joint_reference_token_log_probs"
                    ],
                    response_mask=tensors["response_mask"],
                    valid_row_mask=valid_mask,
                )
                actor_loss = actor_loss + (
                    token_scale
                    * self.joint_training.token_kl_coefficient
                    * token_kl.kl_sum
                )
                token_kl_sum_value += float(token_kl.kl_sum.detach().item())
            if self.scaler is not None:
                self.scaler.scale(actor_loss).backward()
            else:
                actor_loss.backward()
            actor_loss_value += float(actor_loss.detach().item())
            policy_sum_value += float(ppo.policy_loss_sum.detach().item())
            entropy_sum_value += float(ppo.entropy_sum.detach().item())

            hidden = tensors["joint_critic_hidden"].to(
                dtype=next(self.current_joint_critic.parameters()).dtype,
            )
            all_action_values = self.current_joint_critic(hidden)
            critic_terms = selected_action_huber_loss(
                all_action_values,
                tensors["joint_guided_action_ids"],
                tensors["joint_critic_returns"],
                delta=self.joint_training.critic_huber_delta,
                reduction="none",
            )
            critic_loss_sum = (
                critic_terms.per_sample_loss * valid_mask.to(
                    dtype=critic_terms.per_sample_loss.dtype
                )
            ).sum()
            critic_loss = actor_scale * critic_loss_sum
            critic_loss.backward()
            critic_sum_value += float(critic_loss_sum.detach().item())

        actor_grad_norm = self._optimizer_step()
        if not torch.isfinite(actor_grad_norm):
            self.joint_critic_optimizer.zero_grad()
            raise RuntimeError("joint actor gradient norm is non-finite")
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.current_joint_critic.parameters(),
            max_norm=self.joint_training.critic_grad_clip,
        )
        if not torch.isfinite(critic_grad_norm):
            raise RuntimeError("joint critic gradient norm is non-finite")
        self.joint_critic_optimizer.step()
        self.joint_critic_optimizer.zero_grad()
        for key, value in {
            "actor/joint_loss": actor_loss_value,
            "actor/joint_policy_loss_sum": policy_sum_value,
            "actor/guided_entropy_sum": entropy_sum_value,
            "actor/token_kl_sum": token_kl_sum_value,
            "critic/selected_huber_sum": critic_sum_value,
            "actor/grad_norm": float(actor_grad_norm.detach().item()),
            "critic/grad_norm": float(critic_grad_norm.detach().item()),
            "joint/global_valid_turns": float(global_valid.item()),
        }.items():
            metrics.setdefault(key, []).append(value)

    def export_joint_critic_snapshot(
        self,
        *,
        source_step: int,
        contract_id: str,
        score_dtype: str,
    ) -> dict[str, Any]:
        """Export rank-consistency evidence and rank-zero immutable transport."""

        if source_step != self.joint_source_step:
            raise ValueError(
                "joint snapshot source step does not match completed updates"
            )
        if not isinstance(contract_id, str) or not contract_id:
            raise ValueError("joint snapshot contract_id must be non-empty str")
        if self._joint_contract_id is None:
            self._joint_contract_id = contract_id
        elif contract_id != self._joint_contract_id:
            raise ValueError("joint snapshot contract changed within one run")
        from nimloth.training.rl.joint_critic import (
            create_frozen_critic_snapshot,
            export_frozen_critic_snapshot,
        )

        snapshot = create_frozen_critic_snapshot(
            self.current_joint_critic.module,
            source_step=source_step,
            contract_id=contract_id,
            score_dtype=score_dtype,
        )
        state = export_frozen_critic_snapshot(snapshot)
        include_state = self._joint_rank == 0
        return {
            "rank": self._joint_rank,
            "world_size": self._joint_world_size,
            "completed_updates": self._joint_completed_updates,
            "source_step": source_step,
            "snapshot_id": state.snapshot_id,
            "contract_id": state.contract_id,
            "score_dtype": state.score_dtype,
            "optimizer_fingerprint": _optimizer_fingerprint(
                self.joint_critic_optimizer,
            ),
            "snapshot_state": state.to_mapping() if include_state else None,
        }

    def export_joint_checkpoint(
        self,
        *,
        source_step: int,
        contract_id: str,
        score_dtype: str,
    ) -> dict[str, Any]:
        """Return all-rank evidence and one rank-zero restart payload."""

        export = self.export_joint_critic_snapshot(
            source_step=source_step,
            contract_id=contract_id,
            score_dtype=score_dtype,
        )
        fingerprint = export["optimizer_fingerprint"]
        payload = None
        if self._joint_rank == 0:
            payload = {
                "schema": JOINT_ACTOR_CHECKPOINT_SCHEMA,
                "completed_updates": self._joint_completed_updates,
                "source_step": source_step,
                "snapshot_id": export["snapshot_id"],
                "contract_id": export["contract_id"],
                "score_dtype": export["score_dtype"],
                "critic_state": _cpu_clone(
                    self.current_joint_critic.module.state_dict()
                ),
                "critic_optimizer_state": _cpu_clone(
                    self.joint_critic_optimizer.state_dict()
                ),
                "critic_optimizer_fingerprint": fingerprint,
            }
        return {
            **{key: value for key, value in export.items() if key != "snapshot_state"},
            "checkpoint_payload": payload,
        }

    def load_joint_checkpoint(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        """Restore replicated critic and optimizer before the next rollout."""

        if not isinstance(raw, Mapping):
            raise ValueError("joint actor checkpoint must be a mapping")
        fields = {
            "schema",
            "completed_updates",
            "source_step",
            "snapshot_id",
            "contract_id",
            "score_dtype",
            "critic_state",
            "critic_optimizer_state",
            "critic_optimizer_fingerprint",
        }
        if set(raw) != fields or raw["schema"] != JOINT_ACTOR_CHECKPOINT_SCHEMA:
            raise ValueError("joint actor checkpoint schema or fields are invalid")
        completed = _positive_int(raw["completed_updates"], "completed_updates")
        source_step = _nonnegative_int(raw["source_step"], "source_step")
        if source_step != self.joint_training.initial_snapshot_source_step + completed:
            raise ValueError("joint actor checkpoint source step is inconsistent")
        for field in (
            "snapshot_id",
            "contract_id",
            "score_dtype",
            "critic_optimizer_fingerprint",
        ):
            if not isinstance(raw[field], str) or not raw[field]:
                raise ValueError(f"joint actor checkpoint {field} must be non-empty str")
        if (
            self._joint_contract_id is not None
            and raw["contract_id"] != self._joint_contract_id
        ):
            raise ValueError("joint actor checkpoint contract mismatch")
        self._joint_contract_id = raw["contract_id"]
        if raw["score_dtype"] != self.joint_policy.score_dtype:
            raise ValueError("joint actor checkpoint score dtype mismatch")
        self.current_joint_critic.module.load_state_dict(
            raw["critic_state"],
            strict=True,
        )
        self.joint_critic_optimizer.load_state_dict(raw["critic_optimizer_state"])
        self._joint_completed_updates = completed
        actual_optimizer = _optimizer_fingerprint(self.joint_critic_optimizer)
        if actual_optimizer != raw["critic_optimizer_fingerprint"]:
            raise ValueError("joint actor checkpoint optimizer fingerprint mismatch")
        from nimloth.training.rl.joint_critic import create_frozen_critic_snapshot

        snapshot = create_frozen_critic_snapshot(
            self.current_joint_critic.module,
            source_step=source_step,
            contract_id=raw["contract_id"],
            score_dtype=raw["score_dtype"],
        )
        if snapshot.snapshot_id != raw["snapshot_id"]:
            raise ValueError("joint actor checkpoint critic snapshot mismatch")
        dist.barrier()
        return {
            "rank": self._joint_rank,
            "world_size": self._joint_world_size,
            "completed_updates": completed,
            "source_step": source_step,
            "snapshot_id": snapshot.snapshot_id,
            "optimizer_fingerprint": actual_optimizer,
        }


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"joint actor checkpoint {field} must be non-negative int")
    return value


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result < 1:
        raise ValueError(f"joint actor checkpoint {field} must be positive")
    return result


def _score_dtype(name: str) -> torch.dtype:
    values = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
    }
    try:
        return values[name]
    except KeyError as exc:
        raise ValueError(f"unsupported joint score dtype: {name!r}") from exc


def _cpu_clone(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    return value


def _optimizer_fingerprint(optimizer: torch.optim.Optimizer) -> str:
    """Canonical hash used to prove replicated optimizer states agree."""

    digest = hashlib.sha256()
    parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    parameter_index = {id(parameter): index for index, parameter in enumerate(parameters)}
    groups = []
    for group in optimizer.param_groups:
        values = {
            key: value
            for key, value in group.items()
            if key != "params"
        }
        values["params"] = [parameter_index[id(parameter)] for parameter in group["params"]]
        groups.append(values)
    digest.update(
        json.dumps(groups, sort_keys=True, separators=(",", ":"), default=str).encode(
            "utf-8"
        )
    )
    for index, parameter in enumerate(parameters):
        digest.update(str(index).encode("ascii"))
        state = optimizer.state.get(parameter, {})
        for key in sorted(state):
            digest.update(str(key).encode("utf-8"))
            value = state[key]
            if isinstance(value, torch.Tensor):
                tensor = value.detach().contiguous().cpu()
                digest.update(str(tensor.dtype).encode("ascii"))
                digest.update(str(tuple(tensor.shape)).encode("ascii"))
                digest.update(tensor.view(torch.uint8).numpy().tobytes())
            else:
                digest.update(repr(value).encode("utf-8"))
    return f"sha256:{digest.hexdigest()}"


__all__ = ["JOINT_ACTOR_CHECKPOINT_SCHEMA", "JointDataParallelPPOActor"]
