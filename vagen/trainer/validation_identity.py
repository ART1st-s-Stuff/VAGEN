"""Stable identity merge for validation rollout records."""

from collections.abc import Callable, Sequence
from typing import Any


def attach_validation_input_metadata(
    records: list[dict[str, Any]],
    *,
    env_configs: Sequence[dict[str, Any]],
    uids: Sequence[str],
    sources: Sequence[Any],
    input_index_by_env_id: dict[Any, int],
    metadata_fn: Callable[[dict[str, Any], str, Any], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach input metadata by stable environment identity, never list position."""
    input_count = len(env_configs)
    if len(uids) != input_count or len(sources) != input_count:
        raise ValueError(
            "validation input metadata lengths differ: "
            f"env_configs={input_count} uids={len(uids)} sources={len(sources)}"
        )
    if len(records) != input_count:
        raise ValueError(
            f"validation record count differs from inputs: {len(records)} vs {input_count}"
        )

    used_indexes: set[int] = set()
    seen_env_ids: set[Any] = set()
    for record in records:
        env_id = record.get("env_id")
        if env_id in seen_env_ids:
            raise ValueError(f"duplicate validation env_id in records: {env_id!r}")
        seen_env_ids.add(env_id)
        if env_id not in input_index_by_env_id:
            raise ValueError(f"missing stable input identity for env_id={env_id!r}")
        input_index = int(input_index_by_env_id[env_id])
        if not 0 <= input_index < input_count:
            raise ValueError(
                f"input index out of range for env_id={env_id!r}: {input_index}"
            )
        if input_index in used_indexes:
            raise ValueError(f"duplicate validation input index: {input_index}")
        used_indexes.add(input_index)
        record.update(
            metadata_fn(
                env_configs[input_index],
                uids[input_index],
                sources[input_index],
            )
        )

    expected_indexes = set(range(input_count))
    if used_indexes != expected_indexes:
        missing = sorted(expected_indexes - used_indexes)
        raise ValueError(f"validation inputs missing records: {missing[:10]}")
    return records
