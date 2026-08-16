"""Strict summaries for the human-approved K4 canary validation rows."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from typing import Any


def summarize_canary_validation_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_data_sources: Sequence[str],
    expected_rows_per_source: int,
    expected_step: int,
) -> dict[str, Any]:
    """Validate exact held-out coverage and return success/reward aggregates."""

    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise ValueError("canary validation rows must be a sequence")
    sources = _unique_strings(expected_data_sources, "expected_data_sources")
    count = _positive_int(expected_rows_per_source, "expected_rows_per_source")
    step = _nonnegative_int(expected_step, "expected_step")
    expected_total = len(sources) * count
    if len(rows) != expected_total:
        raise ValueError(
            f"canary validation requires exactly {expected_total} rows"
        )

    normalized: list[tuple[str, str, float, int]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"canary validation row {index} must be a mapping")
        missing = {
            "data_source",
            "rollout_sample_id",
            "score",
            "traj_success",
            "step",
        } - set(row)
        if missing:
            raise ValueError(
                f"canary validation row {index} is missing fields: {sorted(missing)}"
            )
        source = _nonempty_string(row["data_source"], "data_source")
        if source not in sources:
            raise ValueError(
                f"canary validation row {index} has unexpected data_source"
            )
        sample_id = _nonempty_string(
            row["rollout_sample_id"],
            "rollout_sample_id",
        )
        if _nonnegative_int(row["step"], "step") != step:
            raise ValueError(
                f"canary validation row {index} has unexpected step"
            )
        score = _finite_float(row["score"], "score")
        success_value = _finite_float(row["traj_success"], "traj_success")
        if success_value not in {0.0, 1.0}:
            raise ValueError("canary validation traj_success must be 0 or 1")
        normalized.append((source, sample_id, score, int(success_value)))

    sample_ids = [item[1] for item in normalized]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("canary validation rollout_sample_id values must be unique")
    source_counts = Counter(item[0] for item in normalized)
    expected_counts = Counter({source: count for source in sources})
    if source_counts != expected_counts:
        raise ValueError(
            "canary validation requires exactly the configured rows per data source"
        )

    by_source: dict[str, dict[str, Any]] = {}
    for source in sources:
        selected = [item for item in normalized if item[0] == source]
        reward_sum = math.fsum(item[2] for item in selected)
        success_count = sum(item[3] for item in selected)
        by_source[source] = {
            "row_count": len(selected),
            "reward_mean": reward_sum / len(selected),
            "success_count": success_count,
            "success_rate": success_count / len(selected),
        }

    reward_sum = math.fsum(item[2] for item in normalized)
    success_count = sum(item[3] for item in normalized)
    return {
        "step": step,
        "row_count": len(normalized),
        "reward_mean": reward_sum / len(normalized),
        "success_count": success_count,
        "success_rate": success_count / len(normalized),
        "by_data_source": by_source,
    }


def _unique_strings(values: Sequence[str], field: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"canary validation {field} must be a sequence")
    result = tuple(_nonempty_string(value, field) for value in values)
    if not result or len(set(result)) != len(result):
        raise ValueError(
            f"canary validation {field} must be non-empty and unique"
        )
    return result


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"canary validation {field} must be a non-empty string")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(
            f"canary validation {field} must be a non-negative integer"
        )
    return int(value)


def _positive_int(value: object, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result == 0:
        raise ValueError(f"canary validation {field} must be positive")
    return result


def _finite_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"canary validation {field} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"canary validation {field} must be finite")
    return result


__all__ = ["summarize_canary_validation_rows"]
