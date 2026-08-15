"""Atomic publication boundary for one replicated joint critic update."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_EXPORT_FIELDS = frozenset(
    {
        "rank",
        "world_size",
        "completed_updates",
        "source_step",
        "snapshot_id",
        "contract_id",
        "score_dtype",
        "optimizer_fingerprint",
        "snapshot_state",
    }
)


def publish_replicated_joint_snapshot(
    *,
    manager: Any,
    rank_exports: Sequence[Mapping[str, Any]],
    expected_world_size: int,
    expected_active_snapshot_id: str,
    expected_active_source_step: int,
    expected_activation_version: int,
) -> dict[str, Any]:
    """Validate every rank before staging and CAS-activating one CPU snapshot."""

    world_size = _positive_int(expected_world_size, "expected_world_size")
    old_source_step = _nonnegative_int(
        expected_active_source_step,
        "expected_active_source_step",
    )
    old_version = _nonnegative_int(
        expected_activation_version,
        "expected_activation_version",
    )
    old_snapshot = _nonempty_string(
        expected_active_snapshot_id,
        "expected_active_snapshot_id",
    )
    exports = list(_plain_sequence(rank_exports, "rank_exports"))
    if len(exports) != world_size:
        raise ValueError(
            f"replicated joint export expected {world_size} ranks, got {len(exports)}"
        )
    canonical = []
    for raw in exports:
        if not isinstance(raw, Mapping):
            raise ValueError("replicated joint rank export must be a mapping")
        missing = _EXPORT_FIELDS - set(raw)
        if missing:
            raise ValueError(f"replicated joint export is missing fields: {sorted(missing)}")
        unexpected = set(raw) - _EXPORT_FIELDS
        if unexpected:
            raise ValueError(
                f"replicated joint export has unexpected fields: {sorted(unexpected)}"
            )
        record = dict(raw)
        record["rank"] = _nonnegative_int(record["rank"], "rank")
        record["world_size"] = _positive_int(record["world_size"], "world_size")
        record["completed_updates"] = _positive_int(
            record["completed_updates"],
            "completed_updates",
        )
        record["source_step"] = _nonnegative_int(record["source_step"], "source_step")
        for field in (
            "snapshot_id",
            "contract_id",
            "score_dtype",
            "optimizer_fingerprint",
        ):
            record[field] = _nonempty_string(record[field], field)
        canonical.append(record)
    ranks = sorted(record["rank"] for record in canonical)
    if ranks != list(range(world_size)):
        raise ValueError(f"replicated joint export ranks are incomplete: {ranks}")
    for record in canonical:
        if record["world_size"] != world_size:
            raise ValueError("replicated joint export world_size mismatch")
    reference = canonical[0]
    if reference["source_step"] != old_source_step + 1:
        raise ValueError("replicated joint snapshot source step must increment by one")
    for field, label in (
        ("completed_updates", "completed update"),
        ("source_step", "source step"),
        ("snapshot_id", "snapshot"),
        ("contract_id", "contract"),
        ("score_dtype", "score dtype"),
        ("optimizer_fingerprint", "optimizer"),
    ):
        if any(record[field] != reference[field] for record in canonical[1:]):
            raise ValueError(f"replicated joint {label} state diverged across ranks")
    state_rows = [record for record in canonical if record["snapshot_state"] is not None]
    if len(state_rows) != 1 or state_rows[0]["rank"] != 0:
        raise ValueError("only replicated joint rank zero may export snapshot state")
    snapshot_state = state_rows[0]["snapshot_state"]
    if not isinstance(snapshot_state, Mapping):
        raise ValueError("replicated joint rank-zero snapshot state must be a mapping")
    source_field = (
        "snapshot_source_step"
        if snapshot_state.get("schema")
        == "vagen_frozen_k4_planner_transport_v1"
        else "source_step"
    )
    for state_field, reference_field in (
        (source_field, "source_step"),
        ("snapshot_id", "snapshot_id"),
        ("contract_id", "contract_id"),
        ("score_dtype", "score_dtype"),
    ):
        if snapshot_state.get(state_field) != reference[reference_field]:
            raise ValueError(
                f"replicated joint snapshot state {state_field} mismatch"
            )

    status = manager.frozen_q_status()
    if not isinstance(status, Mapping):
        raise ValueError("frozen Q manager status must be a mapping")
    if status.get("active_snapshot_id") != old_snapshot:
        raise ValueError("frozen Q active snapshot changed before publication")
    if status.get("active_source_step") != old_source_step:
        raise ValueError("frozen Q active source step changed before publication")
    if status.get("activation_version") != old_version:
        raise ValueError("frozen Q activation version changed before publication")
    if status.get("contract_id") != reference["contract_id"]:
        raise ValueError("frozen Q contract changed before publication")
    if status.get("score_dtype") != reference["score_dtype"]:
        raise ValueError("frozen Q score dtype changed before publication")
    if status.get("open_batch_count") != 0:
        raise ValueError("cannot publish joint snapshot with open rollout batch pins")
    if status.get("staged_snapshot_id") is not None:
        raise ValueError("cannot publish joint snapshot over an existing staged candidate")

    staged = manager.stage_frozen_q_snapshot(
        {
            "new_snapshot_state": dict(snapshot_state),
            "expected_active_snapshot_id": old_snapshot,
            "expected_activation_version": old_version,
        }
    )
    if staged.get("staged_snapshot_id") != reference["snapshot_id"]:
        raise RuntimeError("frozen Q manager staged an unexpected joint snapshot")
    activated = manager.activate_staged_frozen_q_snapshot(
        {
            "staged_snapshot_id": reference["snapshot_id"],
            "expected_active_snapshot_id": old_snapshot,
            "expected_activation_version": old_version,
        }
    )
    if (
        activated.get("active_snapshot_id") != reference["snapshot_id"]
        or activated.get("active_source_step") != reference["source_step"]
        or activated.get("activation_version") != old_version + 1
        or activated.get("staged_snapshot_id") is not None
        or activated.get("open_batch_count") != 0
    ):
        raise RuntimeError("frozen Q manager returned invalid post-activation state")
    return dict(activated)


def _plain_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f"replicated joint {field} must be a plain sequence")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"replicated joint {field} must be non-negative int")
    return value


def _positive_int(value: Any, field: str) -> int:
    result = _nonnegative_int(value, field)
    if result < 1:
        raise ValueError(f"replicated joint {field} must be positive")
    return result


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"replicated joint {field} must be non-empty str")
    return value


__all__ = ["publish_replicated_joint_snapshot"]
