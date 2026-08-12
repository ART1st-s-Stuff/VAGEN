"""Versioned execution facts for future and provisional joint policies."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any

from vagen.joint_policy import GuidedPolicyBehaviorRecord

DECISION_LEDGER_SCHEMA = "vagen_decision_ledger_v1"
GUIDED_DECISION_LEDGER_SCHEMA = "vagen_decision_ledger_v2_frozen_q_guided"

_M1_DECISION_SOURCES = frozenset({"llm_text", "system_fallback"})
_BASE_REQUIRED_FIELDS = frozenset(
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
_GUIDED_REQUIRED_FIELDS = _BASE_REQUIRED_FIELDS | {
    "snapshot_id",
    "contract_id",
    "behavior_record_id",
    "behavior_record",
}


def parse_decision_ledger_enabled(raw: Mapping[str, Any] | None) -> bool:
    """Parse the opt-in feature flag without truthiness coercion."""

    if raw is None:
        return False
    if not isinstance(raw, Mapping):
        raise ValueError("decision_ledger section must be a mapping")
    allowed = {"enabled"}
    unexpected = set(raw) - allowed
    if unexpected:
        raise ValueError(
            f"decision_ledger section has unexpected fields: {sorted(unexpected)}"
        )
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("decision_ledger.enabled must be explicit bool")
    return enabled


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
    """Build one M1 no-concat turn ledger from environment execution facts."""

    if decision_source not in _M1_DECISION_SOURCES:
        raise ValueError(f"unsupported decision source for M1: {decision_source!r}")
    action_names = list(executed_action_names)
    ledger = _base_ledger(
        schema=DECISION_LEDGER_SCHEMA,
        action_space=action_space,
        action_space_names=action_space_names,
        executed_action_ids=executed_action_ids,
        executed_action_names=action_names,
        decision_sources=[decision_source] * len(action_names),
        decision_is_policy_sampled=[False] * len(action_names),
        env_turn_reward=env_turn_reward,
        env_terminated=env_terminated,
        rollout_truncated=rollout_truncated,
        format_valid=format_valid,
    )
    validate_decision_ledger(ledger)
    ledger["env_turn_reward"] = float(env_turn_reward)
    return ledger


def build_guided_decision_ledger(
    *,
    behavior: GuidedPolicyBehaviorRecord,
    env_turn_reward: Real,
    env_terminated: bool,
    rollout_truncated: bool,
    format_valid: bool,
) -> dict[str, Any]:
    """Derive one M2 ledger from a validated rollout behavior record."""

    behavior = GuidedPolicyBehaviorRecord.from_mapping(behavior.to_mapping())
    action_id = behavior.guided_action_id
    ledger = _base_ledger(
        schema=GUIDED_DECISION_LEDGER_SCHEMA,
        action_space=behavior.action_space,
        action_space_names=behavior.action_space_names,
        executed_action_ids=[action_id],
        executed_action_names=[behavior.action_space_names[action_id]],
        decision_sources=["frozen_q_guided"],
        decision_is_policy_sampled=[True],
        env_turn_reward=env_turn_reward,
        env_terminated=env_terminated,
        rollout_truncated=rollout_truncated,
        format_valid=format_valid,
    )
    ledger["snapshot_id"] = behavior.snapshot_id
    ledger["contract_id"] = behavior.contract_id
    ledger["behavior_record_id"] = behavior.record_id()
    ledger["behavior_record"] = behavior.to_mapping()
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
    """Build an M1 ledger from the environment-neutral info contract."""

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
    """Fail closed on malformed or semantically incomplete ledger records."""

    if not isinstance(ledger, Mapping):
        raise ValueError(f"decision ledger must be a mapping, got {type(ledger).__name__}")
    schema = ledger.get("schema")
    if schema == DECISION_LEDGER_SCHEMA:
        required_fields = _BASE_REQUIRED_FIELDS
    elif schema == GUIDED_DECISION_LEDGER_SCHEMA:
        required_fields = _GUIDED_REQUIRED_FIELDS
    else:
        raise ValueError(
            f"unsupported decision ledger schema: {schema!r}; expected "
            f"{DECISION_LEDGER_SCHEMA!r} or {GUIDED_DECISION_LEDGER_SCHEMA!r}"
        )

    missing = required_fields - set(ledger)
    if missing:
        raise ValueError(f"decision ledger is missing fields: {sorted(missing)}")
    unexpected = set(ledger) - required_fields
    if unexpected:
        raise ValueError(f"decision ledger has unexpected fields: {sorted(unexpected)}")

    action_space = ledger["action_space"]
    if not isinstance(action_space, str) or not action_space:
        raise ValueError(f"action_space must be a non-empty string, got {action_space!r}")
    action_space_names = _plain_sequence(ledger["action_space_names"], "action_space_names")
    if not action_space_names or any(
        not isinstance(name, str) or not name for name in action_space_names
    ):
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
    for is_sampled in sampled:
        if not isinstance(is_sampled, bool):
            raise ValueError(
                f"decision_is_policy_sampled must contain bools, got {is_sampled!r}"
            )

    if schema == DECISION_LEDGER_SCHEMA:
        if any(source not in _M1_DECISION_SOURCES for source in sources):
            raise ValueError(f"unsupported decision source for M1: {sources!r}")
        if any(sampled):
            raise ValueError(
                "decision ledger v1 does not define actor-policy sampling or "
                "behavior log-probabilities"
            )
    else:
        if len(action_ids) != 1:
            raise ValueError("guided decision ledger must contain exactly one executed action")
        if sources != ["frozen_q_guided"] or sampled != [True]:
            raise ValueError(
                "guided decision ledger requires one frozen_q_guided policy-owned action"
            )
        for field in ("snapshot_id", "contract_id", "behavior_record_id"):
            if not isinstance(ledger[field], str) or not ledger[field]:
                raise ValueError(f"guided decision ledger {field} must be non-empty")
        behavior = GuidedPolicyBehaviorRecord.from_mapping(ledger["behavior_record"])
        if ledger["snapshot_id"] != behavior.snapshot_id:
            raise ValueError("guided decision ledger snapshot_id does not match behavior")
        if ledger["contract_id"] != behavior.contract_id:
            raise ValueError("guided decision ledger contract_id does not match behavior")
        if ledger["behavior_record_id"] != behavior.record_id():
            raise ValueError("guided decision ledger behavior_record_id does not match behavior")
        if ledger["action_space"] != behavior.action_space or action_space_names != list(
            behavior.action_space_names
        ):
            raise ValueError("guided decision ledger action space does not match behavior")
        if action_ids != [behavior.guided_action_id]:
            raise ValueError("guided decision ledger executed action does not match behavior")

    reward = ledger["env_turn_reward"]
    if (
        isinstance(reward, bool)
        or not isinstance(reward, Real)
        or not math.isfinite(float(reward))
    ):
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


def validate_decision_ledger_reward_rows(
    ledgers: Sequence[Mapping[str, Any]],
    *,
    reward_rows: Sequence[Sequence[Real]],
    response_masks: Sequence[Sequence[int | bool]],
) -> None:
    """Bind ledger reward facts to the exact token reward rows used by PPO."""

    records = list(ledgers)
    rewards = list(reward_rows)
    masks = list(response_masks)
    if len(records) != len(rewards) or len(records) != len(masks):
        raise ValueError(
            "decision ledger, reward row, and response mask batch sizes must match"
        )
    for row_index, (ledger, reward_row, response_mask) in enumerate(
        zip(records, rewards, masks, strict=True)
    ):
        validate_decision_ledger(ledger)
        row = [
            _finite_real(value, f"reward_rows[{row_index}]")
            for value in _plain_sequence(reward_row, f"reward_rows[{row_index}]")
        ]
        mask = _plain_sequence(response_mask, f"response_masks[{row_index}]")
        if len(row) != len(mask):
            raise ValueError(
                f"decision ledger reward row {row_index} does not align with response mask"
            )
        anchor = last_policy_token_index(mask)
        outside_reward = sum(abs(value) for index, value in enumerate(row) if index != anchor)
        if outside_reward != 0.0:
            raise ValueError(
                "decision ledger reward is non-zero outside the last policy-owned "
                f"token for row {row_index}"
            )
        actual_reward = sum(row)
        expected_reward = float(ledger["env_turn_reward"])
        if not math.isclose(actual_reward, expected_reward, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "PPO reward row does not match ledger env_turn_reward: "
                f"row={row_index}, actual={actual_reward}, expected={expected_reward}"
            )


def summarize_decision_ledger_batch(
    ledgers: Sequence[Mapping[str, Any]],
    *,
    expected_batch_size: int,
    allowed_schemas: set[str] | frozenset[str] | None = None,
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
        schema = ledger.get("schema") if isinstance(ledger, Mapping) else None
        if allowed_schemas is not None and schema not in allowed_schemas:
            raise ValueError(
                f"decision ledger schema {schema!r} is not allowed at this "
                "trainer boundary"
            )
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
        "decision_ledger/system_fallback_action_fraction": fallback_actions
        / action_denominator,
        "decision_ledger/policy_sampled_action_fraction": sampled_actions
        / action_denominator,
        "decision_ledger/terminated_turn_fraction": terminated_turns / turn_denominator,
        "decision_ledger/truncated_turn_fraction": truncated_turns / turn_denominator,
        "decision_ledger/format_valid_turn_fraction": format_valid_turns
        / turn_denominator,
    }


def _base_ledger(
    *,
    schema: str,
    action_space: str,
    action_space_names: Sequence[str],
    executed_action_ids: Sequence[int],
    executed_action_names: Sequence[str],
    decision_sources: Sequence[str],
    decision_is_policy_sampled: Sequence[bool],
    env_turn_reward: Real,
    env_terminated: bool,
    rollout_truncated: bool,
    format_valid: bool,
) -> dict[str, Any]:
    return {
        "schema": schema,
        "action_space": action_space,
        "action_space_names": list(action_space_names),
        "executed_action_ids": list(executed_action_ids),
        "executed_action_names": list(executed_action_names),
        "decision_sources": list(decision_sources),
        "decision_is_policy_sampled": list(decision_is_policy_sampled),
        "env_turn_reward": env_turn_reward,
        "env_terminated": env_terminated,
        "rollout_truncated": rollout_truncated,
        "format_valid": format_valid,
    }


def _finite_real(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field} must contain finite real values")
    return float(value)


def _plain_sequence(value: Any, field: str) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a sequence")
    return list(value)


__all__ = [
    "DECISION_LEDGER_SCHEMA",
    "GUIDED_DECISION_LEDGER_SCHEMA",
    "build_decision_ledger",
    "build_decision_ledger_from_env_info",
    "build_guided_decision_ledger",
    "last_policy_token_index",
    "parse_decision_ledger_enabled",
    "summarize_decision_ledger_batch",
    "validate_decision_ledger",
    "validate_decision_ledger_reward_rows",
]
