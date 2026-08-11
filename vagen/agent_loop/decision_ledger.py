"""Versioned execution facts for a future joint action policy.

Milestone M1 intentionally records no actor logits or behavior probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

DECISION_LEDGER_SCHEMA = "vagen_decision_ledger_v1"

_ALLOWED_DECISION_SOURCES = frozenset({"llm_text", "system_fallback"})
_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "action_space",
        "action_space_names",
        "executed_action_ids",
        "executed_action_names",
        "decision_sources",
        "decision_is_policy_sampled",
        "env_turn_reward",
        "env_terminated",
        "rollout_truncated",
        "format_valid",
    }
)


def build_decision_ledger(
    *,
    action_space: str,
    action_space_names: Sequence[str],
    executed_action_ids: Sequence[int],
    executed_action_names: Sequence[str],
    decision_source: str,
    env_turn_reward: Real,
    env_terminated: bool,
    rollout_truncated: bool,
    format_valid: bool,
) -> dict[str, Any]:
    """Build one no-concat turn ledger from actions the environment executed."""

    if decision_source not in _ALLOWED_DECISION_SOURCES:
        raise ValueError(f"unsupported decision source for M1: {decision_source!r}")

    ledger = {
        "schema": DECISION_LEDGER_SCHEMA,
        "action_space": action_space,
        "action_space_names": list(action_space_names),
        "executed_action_ids": list(executed_action_ids),
        "executed_action_names": list(executed_action_names),
        "decision_sources": [decision_source] * len(executed_action_names),
        # M1 has no actor sampler. A later schema must add exact behavior
        # probabilities before any action can claim policy ownership.
        "decision_is_policy_sampled": [False] * len(executed_action_names),
        "env_turn_reward": env_turn_reward,
        "env_terminated": env_terminated,
        "rollout_truncated": rollout_truncated,
        "format_valid": format_valid,
    }
    validate_decision_ledger(ledger)
    ledger["env_turn_reward"] = float(env_turn_reward)
    return ledger


def build_decision_ledger_from_env_info(
    info: Mapping[str, Any],
    *,
    env_turn_reward: Real,
    env_terminated: bool,
    rollout_truncated: bool,
) -> dict[str, Any]:
    """Build a ledger from the environment-neutral M1 info contract."""

    action_space = info.get("action_space")
    action_space_names = info.get("action_space_names")
    action_ids = info.get("executed_action_ids")
    action_names = info.get("executed_action_names")
    if not isinstance(action_space, str) or not action_space:
        raise ValueError("decision ledger requires a non-empty environment info action_space")
    for field, value in (
        ("action_space_names", action_space_names),
        ("executed_action_ids", action_ids),
        ("executed_action_names", action_names),
    ):
        if not isinstance(value, list):
            raise ValueError(f"decision ledger requires environment info list {field}")

    format_correct = info.get("format_correct")
    if not isinstance(format_correct, bool):
        raise ValueError(f"environment info format_correct must be bool, got {format_correct!r}")
    fallback_used = info.get("planner_fallback_used")
    if not isinstance(fallback_used, bool):
        raise ValueError(
            "environment info planner_fallback_used must be bool, "
            f"got {fallback_used!r}"
        )

    return build_decision_ledger(
        action_space=action_space,
        action_space_names=action_space_names,
        executed_action_ids=action_ids,
        executed_action_names=action_names,
        decision_source="system_fallback" if fallback_used else "llm_text",
        env_turn_reward=env_turn_reward,
        env_terminated=env_terminated,
        rollout_truncated=rollout_truncated,
        format_valid=format_correct,
    )


def validate_decision_ledger(ledger: Mapping[str, Any]) -> None:
    """Fail closed on malformed or prematurely policy-owned M1 records."""

    if not isinstance(ledger, Mapping):
        raise ValueError(f"decision ledger must be a mapping, got {type(ledger).__name__}")

    missing = _REQUIRED_FIELDS - set(ledger)
    if missing:
        raise ValueError(f"decision ledger is missing fields: {sorted(missing)}")
    unexpected = set(ledger) - _REQUIRED_FIELDS
    if unexpected:
        raise ValueError(f"decision ledger has unexpected fields: {sorted(unexpected)}")

    schema = ledger["schema"]
    if schema != DECISION_LEDGER_SCHEMA:
        raise ValueError(
            f"unsupported decision ledger schema: {schema!r}; expected {DECISION_LEDGER_SCHEMA!r}"
        )

    action_space = ledger["action_space"]
    if not isinstance(action_space, str) or not action_space:
        raise ValueError(f"action_space must be a non-empty string, got {action_space!r}")
    action_space_names = _plain_sequence(ledger["action_space_names"], "action_space_names")
    if not action_space_names or any(not isinstance(name, str) or not name for name in action_space_names):
        raise ValueError("action_space_names must contain non-empty strings")
    if len(set(action_space_names)) != len(action_space_names):
        raise ValueError("action_space_names must be unique")

    action_ids = _plain_sequence(ledger["executed_action_ids"], "executed_action_ids")
    action_names = _plain_sequence(ledger["executed_action_names"], "executed_action_names")
    sources = _plain_sequence(ledger["decision_sources"], "decision_sources")
    sampled = _plain_sequence(
        ledger["decision_is_policy_sampled"],
        "decision_is_policy_sampled",
    )
    lengths = {len(action_ids), len(action_names), len(sources), len(sampled)}
    if len(lengths) != 1:
        raise ValueError("decision ledger action fields must have the same length")

    for action_id, action_name in zip(action_ids, action_names, strict=True):
        if isinstance(action_id, bool) or not isinstance(action_id, int) or action_id < 0:
            raise ValueError(f"invalid executed action id: {action_id!r}")
        if not isinstance(action_name, str) or not action_name:
            raise ValueError(f"invalid executed action name: {action_name!r}")
        if action_id >= len(action_space_names) or action_space_names[action_id] != action_name:
            raise ValueError(
                f"executed action ({action_id}, {action_name!r}) does not match action space "
                f"{action_space!r}"
            )
    for source in sources:
        if source not in _ALLOWED_DECISION_SOURCES:
            raise ValueError(f"unsupported decision source for M1: {source!r}")
    for is_sampled in sampled:
        if not isinstance(is_sampled, bool):
            raise ValueError(f"decision_is_policy_sampled must contain bools, got {is_sampled!r}")
    if any(sampled):
        raise ValueError(
            "decision ledger v1 does not define actor-policy sampling or behavior log-probabilities"
        )

    reward = ledger["env_turn_reward"]
    if isinstance(reward, bool) or not isinstance(reward, Real) or not math.isfinite(float(reward)):
        raise ValueError(f"env_turn_reward must be finite, got {reward!r}")

    for field in ("env_terminated", "rollout_truncated", "format_valid"):
        if not isinstance(ledger[field], bool):
            raise ValueError(f"{field} must be bool, got {ledger[field]!r}")
    if ledger["env_terminated"] and ledger["rollout_truncated"]:
        raise ValueError("a decision ledger turn cannot be both terminated and truncated")


def last_policy_token_index(response_mask: Sequence[int | bool]) -> int:
    """Return the last sampled-token index, excluding injected suffix tokens."""

    mask = _plain_sequence(response_mask, "response_mask")
    last_index = -1
    for index, value in enumerate(mask):
        if isinstance(value, bool):
            is_policy_token = value
        elif isinstance(value, int) and value in (0, 1):
            is_policy_token = value == 1
        else:
            raise ValueError(f"response_mask must contain only 0/1 values, got {value!r}")
        if is_policy_token:
            last_index = index
    if last_index < 0:
        raise ValueError("cannot anchor turn reward: response has no policy-owned token")
    return last_index


def summarize_decision_ledger_batch(
    ledgers: Sequence[Mapping[str, Any]],
    *,
    expected_batch_size: int,
) -> dict[str, float]:
    """Validate a complete no-concat batch and return ownership diagnostics."""

    records = list(ledgers)
    if len(records) != expected_batch_size:
        raise ValueError(
            f"expected {expected_batch_size} decision ledgers, got {len(records)}"
        )
    if expected_batch_size <= 0:
        raise ValueError("decision ledger batch must be non-empty")

    total_actions = 0
    turns_with_actions = 0
    fallback_actions = 0
    sampled_actions = 0
    terminated_turns = 0
    truncated_turns = 0
    format_valid_turns = 0

    for ledger in records:
        validate_decision_ledger(ledger)
        action_count = len(ledger["executed_action_ids"])
        total_actions += action_count
        turns_with_actions += int(action_count > 0)
        fallback_actions += sum(
            source == "system_fallback" for source in ledger["decision_sources"]
        )
        sampled_actions += sum(ledger["decision_is_policy_sampled"])
        terminated_turns += int(ledger["env_terminated"])
        truncated_turns += int(ledger["rollout_truncated"])
        format_valid_turns += int(ledger["format_valid"])

    action_denominator = max(total_actions, 1)
    turn_denominator = float(expected_batch_size)
    return {
        "decision_ledger/turn_coverage": 1.0,
        "decision_ledger/executed_action_count_mean": total_actions / turn_denominator,
        "decision_ledger/action_turn_coverage": turns_with_actions / turn_denominator,
        "decision_ledger/system_fallback_action_fraction": fallback_actions / action_denominator,
        "decision_ledger/policy_sampled_action_fraction": sampled_actions / action_denominator,
        "decision_ledger/terminated_turn_fraction": terminated_turns / turn_denominator,
        "decision_ledger/truncated_turn_fraction": truncated_turns / turn_denominator,
        "decision_ledger/format_valid_turn_fraction": format_valid_turns / turn_denominator,
    }


def _plain_sequence(value: Any, field: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    return list(value)


__all__ = [
    "DECISION_LEDGER_SCHEMA",
    "build_decision_ledger",
    "build_decision_ledger_from_env_info",
    "last_policy_token_index",
    "summarize_decision_ledger_batch",
    "validate_decision_ledger",
]
