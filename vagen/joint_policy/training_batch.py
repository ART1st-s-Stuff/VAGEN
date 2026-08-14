"""Strict DataProto compilation for one guided joint update."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .contract import GuidedPolicyBehaviorRecord
from .terminal_state import TerminalStateTrace
from .training_contract import (
    JointTrainingConfig,
    JointTrainingTargets,
    compile_outcome_returns_and_frozen_v_gae,
)


def prepare_joint_training_batch(
    batch: Any,
    *,
    config: JointTrainingConfig,
) -> JointTrainingTargets:
    """Revalidate rollout evidence and add tensor-only actor/critic targets."""

    if not isinstance(config, JointTrainingConfig):
        raise TypeError("joint batch preparation requires JointTrainingConfig")
    size = len(batch)
    if size < 1:
        raise ValueError("joint training batch must be non-empty")
    required_non_tensor = {
        "group_idx",
        "traj_idx",
        "guided_turn_index",
        "rollout_stop_reason",
        "decision_ledger",
        "policy_state",
        "policy_response_trace",
        "joint_policy_batch_pin",
        "terminal_state_trace",
    }
    missing = required_non_tensor - set(batch.non_tensor_batch)
    if missing:
        raise ValueError(
            f"joint training batch is missing rollout evidence: {sorted(missing)}"
        )
    if "responses" not in batch.batch or "response_mask" not in batch.batch:
        raise ValueError("joint training batch requires responses and response_mask")

    rows = []
    behaviors = []
    hidden_rows = []
    action_tables = []
    prior_tokens = []
    prior_indices = []
    pins = []
    trajectory_final_rows: dict[tuple[str, int], int] = {}
    for index in range(size):
        ledger = batch.non_tensor_batch["decision_ledger"][index]
        stop_reason = batch.non_tensor_batch["rollout_stop_reason"][index]
        row = {
            "group_idx": batch.non_tensor_batch["group_idx"][index],
            "traj_idx": _int_value(
                batch.non_tensor_batch["traj_idx"][index],
                "traj_idx",
            ),
            "guided_turn_index": _int_value(
                batch.non_tensor_batch["guided_turn_index"][index],
                "guided_turn_index",
            ),
            "rollout_stop_reason": stop_reason,
            "decision_ledger": ledger,
        }
        rows.append(row)
        behavior = GuidedPolicyBehaviorRecord.from_mapping(
            ledger["behavior_record"]
        )
        behaviors.append(behavior)
        state = _policy_state(batch.non_tensor_batch["policy_state"][index])
        if state["action_token_ids"] != list(behavior.action_token_ids):
            raise ValueError("joint training policy state action token table mismatch")
        if len(state["latent_token_ids"]) != config.critic_grid_tokens:
            raise ValueError("joint training policy latent token count mismatch")
        if (
            len(state["action_logits"]) != config.critic_action_count
            or any(not math.isfinite(float(value)) for value in state["action_logits"])
            or any(
                not _score_close(actual, expected, behavior.policy_config.score_dtype)
                for actual, expected in zip(
                    state["action_logits"],
                    behavior.prior_logits,
                    strict=True,
                )
            )
        ):
            raise ValueError("joint training policy action logits mismatch behavior")
        hidden = state["latent_hidden"]
        if (
            len(hidden) != config.critic_grid_tokens
            or any(
                len(values) != config.critic_qwen_hidden_dim for values in hidden
            )
            or any(
                not math.isfinite(float(value))
                for values in hidden
                for value in values
            )
        ):
            raise ValueError(
                "joint training policy hidden shape or values mismatch critic config"
            )
        hidden_rows.append(hidden)
        action_tables.append(list(behavior.action_token_ids))
        prior_tokens.append(behavior.prior_token_id)
        prior_indices.append(behavior.prior_response_idx)
        trace = _response_trace(
            batch.non_tensor_batch["policy_response_trace"][index]
        )
        if (
            trace["request_id"] != state["request_id"]
            or trace["generation_id"] != state["generation_id"]
        ):
            raise ValueError("joint training response and policy state identity mismatch")
        if behavior.prior_response_idx >= len(trace["response_ids"]):
            raise ValueError("joint training prior response index is outside trace")
        if trace["response_ids"][behavior.prior_response_idx] != behavior.prior_token_id:
            raise ValueError("joint training prior response token identity mismatch")
        response = batch.batch["responses"][index]
        response_mask = batch.batch["response_mask"][index]
        trace_width = len(trace["response_ids"])
        if not torch.equal(
            response[:trace_width].detach().cpu(),
            torch.tensor(trace["response_ids"], dtype=response.dtype),
        ):
            raise ValueError("joint training DataProto response IDs mismatch trace")
        if not torch.equal(
            response_mask[:trace_width].detach().cpu().to(dtype=torch.long),
            torch.tensor(trace["response_mask"], dtype=torch.long),
        ):
            raise ValueError("joint training DataProto response mask mismatch trace")

        pin = _pin(batch.non_tensor_batch["joint_policy_batch_pin"][index])
        if pin["snapshot_id"] != behavior.snapshot_id or pin["contract_id"] != behavior.contract_id:
            raise ValueError("joint training batch pin does not match behavior")
        pins.append(pin)
        identity = (str(row["group_idx"]), row["traj_idx"])
        if stop_reason != "continue":
            if identity in trajectory_final_rows:
                raise ValueError("joint training trajectory has multiple final rows")
            trajectory_final_rows[identity] = index

    targets = compile_outcome_returns_and_frozen_v_gae(rows, config=config)
    if any(
        pin != pins[0]
        for pin in pins[1:]
    ):
        raise ValueError("joint training batch must share one manager batch pin")
    for index, row in enumerate(rows):
        terminal_raw = batch.non_tensor_batch["terminal_state_trace"][index]
        final = row["rollout_stop_reason"] != "continue"
        if final:
            terminal = TerminalStateTrace.from_mapping(terminal_raw)
            if terminal.rollout_stop_reason != row["rollout_stop_reason"]:
                raise ValueError("terminal state trace outcome mismatch")
        elif terminal_raw is not None:
            raise ValueError("non-final joint turn cannot contain terminal state trace")

    action_count = config.critic_action_count
    if any(len(table) != action_count or table != action_tables[0] for table in action_tables):
        raise ValueError("joint training batch must share one action token table")
    batch.batch["joint_action_token_ids"] = torch.tensor(
        action_tables,
        dtype=torch.long,
    )
    batch.batch["joint_prior_token_ids"] = torch.tensor(prior_tokens, dtype=torch.long)
    batch.batch["joint_prior_response_indices"] = torch.tensor(
        prior_indices,
        dtype=torch.long,
    )
    batch.batch["joint_guided_action_ids"] = torch.tensor(
        targets.executed_action_ids,
        dtype=torch.long,
    )
    batch.batch["joint_behavior_guided_log_probs"] = torch.tensor(
        [behavior.behavior_guided_logprob for behavior in behaviors],
        dtype=torch.float32,
    )
    batch.batch["joint_frozen_all_action_q"] = torch.tensor(
        [behavior.frozen_all_action_q for behavior in behaviors],
        dtype=torch.float32,
    )
    batch.batch["joint_advantages"] = torch.tensor(
        targets.advantages,
        dtype=torch.float32,
    )
    batch.batch["joint_critic_hidden"] = torch.tensor(
        np.asarray(hidden_rows, dtype=np.float32),
        dtype=torch.float32,
    )
    batch.batch["joint_critic_returns"] = torch.tensor(
        targets.discounted_returns,
        dtype=torch.float32,
    )
    batch.batch["joint_valid_mask"] = torch.ones(size, dtype=torch.bool)
    batch.meta_info["joint_snapshot_id"] = targets.snapshot_id
    batch.meta_info["joint_contract_id"] = targets.contract_id
    batch.meta_info["joint_snapshot_source_step"] = pins[0]["snapshot_source_step"]
    batch.meta_info["joint_activation_version"] = pins[0]["activation_version"]
    return targets


def mark_joint_padding_invalid(batch: Any, pad_size: int) -> None:
    """Clear the joint loss mask for rows duplicated by DataProto padding."""

    if isinstance(pad_size, bool) or not isinstance(pad_size, int) or pad_size < 0:
        raise ValueError("joint padding size must be non-negative int")
    if "joint_valid_mask" not in batch.batch:
        raise ValueError("joint padding requires a precompiled valid mask")
    if pad_size > len(batch):
        raise ValueError("joint padding size exceeds batch")
    if pad_size:
        batch.batch["joint_valid_mask"][-pad_size:] = False


def _policy_state(raw: Any) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("joint training policy_state must be mapping")
    required = {
        "schema",
        "request_id",
        "generation_id",
        "latent_token_ids",
        "action_start_token_id",
        "action_token_ids",
        "latent_hidden",
        "action_logits",
    }
    if set(raw) != required or raw["schema"] != "nimloth_policy_state_v2":
        raise ValueError("joint training policy_state schema or fields are invalid")
    if (
        not isinstance(raw["request_id"], str)
        or not raw["request_id"]
        or not isinstance(raw["generation_id"], str)
        or not raw["generation_id"]
        or raw["request_id"] == raw["generation_id"]
    ):
        raise ValueError("joint training policy state identity is invalid")
    for field in ("latent_token_ids", "action_token_ids"):
        values = raw[field]
        if (
            isinstance(values, (str, bytes))
            or not isinstance(values, (list, tuple))
            or not values
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in values
            )
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"joint training policy state {field} is invalid")
    if (
        isinstance(raw["action_start_token_id"], bool)
        or not isinstance(raw["action_start_token_id"], int)
        or raw["action_start_token_id"] < 0
        or raw["action_start_token_id"] in raw["latent_token_ids"]
    ):
        raise ValueError("joint training policy action_start_token_id is invalid")
    for field in ("latent_hidden", "action_logits"):
        if isinstance(raw[field], (str, bytes)) or not isinstance(
            raw[field],
            (list, tuple),
        ):
            raise ValueError(f"joint training policy state {field} must be sequence")
    return raw


def _response_trace(raw: Any) -> Mapping[str, Any]:
    from nimloth.training.rl.joint_behavior import NimlothPolicyResponseTrace

    trace = NimlothPolicyResponseTrace.from_mapping(raw)
    return trace.to_mapping()


def _pin(raw: Any) -> dict[str, Any]:
    from nimloth.training.rl.joint_frozen_q_owner import FrozenQBatchPin

    return FrozenQBatchPin.from_mapping(raw).to_mapping()


def _score_close(actual: Any, expected: float, dtype: str) -> bool:
    try:
        value = float(actual)
    except (TypeError, ValueError):
        return False
    tolerance = {
        "float64": 1e-12,
        "float32": 1e-6,
        "bfloat16": 1e-2,
    }[dtype]
    return math.isfinite(value) and math.isclose(
        value,
        float(expected),
        rel_tol=0.0,
        abs_tol=tolerance * max(1.0, abs(float(expected))),
    )


def _int_value(value: Any, field: str) -> int:
    if isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"joint training {field} must be non-negative int")
    return value


__all__ = ["mark_joint_padding_invalid", "prepare_joint_training_batch"]
