"""Strict DataProto compilation for one guided joint update."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .contract import GuidedPolicyBehaviorRecord
from .planning_contract import K4MCTSGuidedBehaviorRecord
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
    policy_states = []
    hidden_rows = []
    guidance_rows = []
    direct_q_rows = []
    policy_implementations = []
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
        behavior = _guided_behavior_record(ledger)
        behaviors.append(behavior)
        if isinstance(behavior, K4MCTSGuidedBehaviorRecord):
            guidance_rows.append(list(behavior.planner_root_mean_values))
            direct_q_rows.append(list(behavior.direct_all_action_q))
        else:
            guidance_rows.append(list(behavior.frozen_all_action_q))
            direct_q_rows.append(list(behavior.frozen_all_action_q))
        policy_implementations.append(behavior.policy_config.implementation)
        state = _policy_state(
            batch.non_tensor_batch["policy_state"][index],
            k4=isinstance(behavior, K4MCTSGuidedBehaviorRecord),
        )
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
        policy_states.append(state)
        hidden_rows.append(hidden)
        action_tables.append(list(behavior.action_token_ids))
        prior_tokens.append(behavior.prior_token_id)
        prior_indices.append(behavior.prior_response_idx)
        pin = _pin(batch.non_tensor_batch["joint_policy_batch_pin"][index])
        if pin["snapshot_id"] != behavior.snapshot_id or pin["contract_id"] != behavior.contract_id:
            raise ValueError("joint training batch pin does not match behavior")
        if isinstance(behavior, K4MCTSGuidedBehaviorRecord):
            _validate_k4_policy_state(
                state,
                behavior=behavior,
                pin=pin,
            )
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

        pins.append(pin)
        identity = (str(row["group_idx"]), row["traj_idx"])
        if stop_reason != "continue":
            if identity in trajectory_final_rows:
                raise ValueError("joint training trajectory has multiple final rows")
            trajectory_final_rows[identity] = index

    targets = compile_outcome_returns_and_frozen_v_gae(rows, config=config)
    if len(set(policy_implementations)) != 1:
        raise ValueError("joint training batch cannot mix policy implementations")
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
    batch.batch["joint_frozen_direct_all_action_q"] = torch.tensor(
        direct_q_rows,
        dtype=torch.float32,
    )
    if isinstance(behaviors[0], K4MCTSGuidedBehaviorRecord):
        batch.batch["joint_frozen_planner_root_mean_values"] = torch.tensor(
            guidance_rows,
            dtype=torch.float32,
        )
    else:
        batch.batch["joint_frozen_all_action_q"] = torch.tensor(
            guidance_rows,
            dtype=torch.float32,
        )
    batch.batch["joint_advantages"] = torch.tensor(
        targets.advantages,
        dtype=torch.float32,
    )
    batch.batch["joint_frozen_state_values"] = torch.tensor(
        targets.frozen_state_values,
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
    if isinstance(behaviors[0], K4MCTSGuidedBehaviorRecord):
        _compile_k4_future_hidden(
            batch,
            rows=rows,
            policy_states=policy_states,
            executed_action_ids=targets.executed_action_ids,
            prediction_horizon=behaviors[0].policy_config.planning_horizon,
        )
    batch.meta_info["joint_snapshot_id"] = targets.snapshot_id
    batch.meta_info["joint_contract_id"] = targets.contract_id
    batch.meta_info["joint_snapshot_source_step"] = pins[0]["snapshot_source_step"]
    batch.meta_info["joint_activation_version"] = pins[0]["activation_version"]
    batch.meta_info["joint_policy_implementation"] = policy_implementations[0]
    return targets


def joint_data_metrics(batch: Any) -> dict[str, float]:
    """Row-level metrics that do not pretend joint targets are token values."""

    required = {
        "joint_valid_mask",
        "joint_advantages",
        "joint_critic_returns",
        "joint_frozen_state_values",
        "token_level_scores",
        "token_level_rewards",
        "response_mask",
    }
    missing = required - set(batch.batch.keys())
    if missing:
        raise ValueError(f"joint data metrics missing tensors: {sorted(missing)}")
    valid = batch.batch["joint_valid_mask"].to(dtype=torch.bool)
    if int(valid.sum().item()) < 1:
        raise ValueError("joint data metrics require at least one valid turn")
    values = {
        "joint/advantage": batch.batch["joint_advantages"][valid],
        "joint/critic_return": batch.batch["joint_critic_returns"][valid],
        "joint/frozen_state_value": batch.batch["joint_frozen_state_values"][valid],
        "joint/sequence_score": batch.batch["token_level_scores"][valid].sum(-1),
        "joint/sequence_reward": batch.batch["token_level_rewards"][valid].sum(-1),
        "joint/response_length": batch.batch["response_mask"][valid].sum(-1).float(),
    }
    metrics: dict[str, float] = {}
    for prefix, tensor in values.items():
        metrics[f"{prefix}/mean"] = float(tensor.float().mean().detach().item())
        metrics[f"{prefix}/min"] = float(tensor.min().detach().item())
        metrics[f"{prefix}/max"] = float(tensor.max().detach().item())
    metrics["joint/valid_turn_count"] = float(valid.sum().item())
    return metrics


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


def _policy_state(raw: Any, *, k4: bool = False) -> Mapping[str, Any]:
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
    expected_schema = "nimloth_policy_state_v2"
    if k4:
        required.add("frozen_k4_planning")
        expected_schema = "nimloth_policy_state_k4_mcts_v1"
    if set(raw) != required or raw["schema"] != expected_schema:
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


def _guided_behavior_record(
    ledger: Mapping[str, Any],
) -> GuidedPolicyBehaviorRecord | K4MCTSGuidedBehaviorRecord:
    if ledger.get("schema") == "vagen_decision_ledger_v3_k4_mcts_guided":
        return K4MCTSGuidedBehaviorRecord.from_mapping(ledger["behavior_record"])
    return GuidedPolicyBehaviorRecord.from_mapping(ledger["behavior_record"])


def _validate_k4_policy_state(
    state: Mapping[str, Any],
    *,
    behavior: K4MCTSGuidedBehaviorRecord,
    pin: Mapping[str, Any],
) -> None:
    from nimloth.training.rl.joint_planning_scoring import (
        k4_scoring_record_from_policy_state,
    )

    score = k4_scoring_record_from_policy_state(
        state,
        expected_request_id=state["request_id"],
        expected_generation_id=state["generation_id"],
        expected_latent_token_ids=state["latent_token_ids"],
        expected_action_start_token_id=state["action_start_token_id"],
        expected_action_token_ids=behavior.action_token_ids,
        expected_snapshot_id=behavior.snapshot_id,
        expected_snapshot_source_step=pin["snapshot_source_step"],
        expected_contract_id=behavior.contract_id,
        expected_activation_version=pin["activation_version"],
        expected_score_dtype=behavior.policy_config.score_dtype,
        expected_planning_horizon=behavior.policy_config.planning_horizon,
        expected_mcts_num_simulations=(
            behavior.policy_config.mcts_num_simulations
        ),
        expected_mcts_exploration_constant=(
            behavior.policy_config.mcts_exploration_constant
        ),
    )
    dtype = behavior.policy_config.score_dtype
    if any(
        not _score_close(actual, expected, dtype)
        for actual, expected in zip(
            score.direct_all_action_q,
            behavior.direct_all_action_q,
            strict=True,
        )
    ):
        raise ValueError("joint training K4 direct Q mismatch behavior")
    if any(
        not _score_close(actual, expected, dtype)
        for actual, expected in zip(
            score.planner_root_mean_values,
            behavior.planner_root_mean_values,
            strict=True,
        )
    ):
        raise ValueError("joint training K4 planner root means mismatch behavior")
    if score.planner_root_visit_counts != behavior.planner_root_visit_counts:
        raise ValueError("joint training K4 planner root visits mismatch behavior")


def _compile_k4_future_hidden(
    batch: Any,
    *,
    rows: list[dict[str, Any]],
    policy_states: list[Mapping[str, Any]],
    executed_action_ids: tuple[int, ...],
    prediction_horizon: int,
) -> None:
    size = len(rows)
    if len(policy_states) != size or len(executed_action_ids) != size:
        raise ValueError("K4 WM compiler input rows do not align")
    first_hidden = policy_states[0]["latent_hidden"]
    grid_tokens = len(first_hidden)
    hidden_dim = len(first_hidden[0])
    future_hidden = np.zeros(
        (size, prediction_horizon, grid_tokens, hidden_dim),
        dtype=np.float32,
    )
    future_actions = np.zeros(
        (size, prediction_horizon),
        dtype=np.int64,
    )
    future_valid = np.zeros(
        (size, prediction_horizon),
        dtype=np.bool_,
    )
    trajectories: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(rows):
        identity = (str(row["group_idx"]), int(row["traj_idx"]))
        trajectories.setdefault(identity, []).append(index)
    window_count = 0
    for identity, indices in trajectories.items():
        indices.sort(key=lambda index: rows[index]["guided_turn_index"])
        final_index = indices[-1]
        terminal = TerminalStateTrace.from_mapping(
            batch.non_tensor_batch["terminal_state_trace"][final_index]
        )
        terminal_hidden = np.asarray(terminal.latent_hidden, dtype=np.float32)
        if terminal_hidden.shape != (grid_tokens, hidden_dim):
            raise ValueError(
                "K4 WM terminal hidden shape does not match policy state: "
                f"trajectory={identity}, shape={terminal_hidden.shape}"
            )
        state_sequence = [
            np.asarray(policy_states[index]["latent_hidden"], dtype=np.float32)
            for index in indices
        ] + [terminal_hidden]
        if any(value.shape != (grid_tokens, hidden_dim) for value in state_sequence):
            raise ValueError("K4 WM policy hidden shapes are inconsistent")
        for position, source_index in enumerate(indices):
            available = min(prediction_horizon, len(indices) - position)
            window_count += available
            for offset in range(available):
                action_index = indices[position + offset]
                future_actions[source_index, offset] = executed_action_ids[action_index]
                future_hidden[source_index, offset] = state_sequence[position + offset + 1]
                future_valid[source_index, offset] = True
    batch.batch["joint_wm_future_hidden"] = torch.from_numpy(future_hidden)
    batch.batch["joint_wm_future_action_ids"] = torch.from_numpy(future_actions)
    batch.batch["joint_wm_future_valid_mask"] = torch.from_numpy(future_valid)
    batch.meta_info["joint_wm_window_count"] = window_count


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


__all__ = [
    "joint_data_metrics",
    "mark_joint_padding_invalid",
    "prepare_joint_training_batch",
]
