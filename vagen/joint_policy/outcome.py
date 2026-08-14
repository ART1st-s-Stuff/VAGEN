"""Learning outcome classification for guided environment turns."""

from __future__ import annotations

from typing import Any


def classify_rollout_stop_reason(
    *,
    success: bool,
    env_terminated: bool,
    turn_count: int,
    max_turns: int,
    response_limit_exhausted: bool,
) -> str:
    """Separate task outcomes from invalid infrastructure truncation."""

    for field, value in (
        ("success", success),
        ("env_terminated", env_terminated),
        ("response_limit_exhausted", response_limit_exhausted),
    ):
        if not isinstance(value, bool):
            raise ValueError(f"rollout outcome {field} must be bool")
    turns = _positive_int(turn_count, "turn_count")
    limit = _positive_int(max_turns, "max_turns")
    if turns > limit:
        raise ValueError("rollout turn_count cannot exceed max_turns")
    if success:
        if not env_terminated:
            raise ValueError("successful rollout outcome must be environment terminal")
        return "success"
    if env_terminated:
        return "environment_failure"
    if turns == limit:
        return "task_failure"
    if response_limit_exhausted:
        return "infrastructure_truncation"
    return "continue"


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"rollout outcome {field} must be positive int")
    return value


__all__ = ["classify_rollout_stop_reason"]
