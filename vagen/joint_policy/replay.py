"""Pure replay of rollout-persisted frozen-Q guided behavior."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .contract import GuidedPolicyBehaviorRecord
from .torch_policy import frozen_q_guided_log_probs


def replay_guided_behavior_log_probs(
    current_prior_logits: Any,
    behavior_records: Sequence[GuidedPolicyBehaviorRecord],
    *,
    expected_contract_id: str,
    expected_snapshot_id: str,
) -> dict[str, Any]:
    """Replay selected guided-action log-probs without recomputing behavior Q.

    ``current_prior_logits`` is the only trainable input. Frozen Q values come
    exclusively from the immutable rollout records and are detached again by
    :func:`frozen_q_guided_log_probs`.
    """

    import torch

    if not isinstance(expected_contract_id, str) or not expected_contract_id:
        raise ValueError("joint replay expected_contract_id must be non-empty")
    if not isinstance(expected_snapshot_id, str) or not expected_snapshot_id:
        raise ValueError("joint replay expected_snapshot_id must be non-empty")
    if isinstance(behavior_records, (str, bytes)) or not isinstance(
        behavior_records, Sequence
    ):
        raise ValueError("joint replay behavior_records must be a sequence")
    if not behavior_records:
        raise ValueError("joint replay behavior_records must be non-empty")
    if not isinstance(current_prior_logits, torch.Tensor):
        raise ValueError("joint replay current_prior_logits must be a torch Tensor")
    if current_prior_logits.ndim != 2:
        raise ValueError(
            "joint replay current_prior_logits must have shape [batch, actions]"
        )

    records: list[GuidedPolicyBehaviorRecord] = []
    for index, record in enumerate(behavior_records):
        if not isinstance(record, GuidedPolicyBehaviorRecord):
            raise ValueError(
                "joint replay behavior record must be GuidedPolicyBehaviorRecord: "
                f"index={index}, type={type(record).__name__}"
            )
        # Rebuild instead of trusting an already-constructed dataclass. This
        # catches malformed records produced with dataclasses.replace or unsafe
        # deserialization before any replay tensor is created.
        records.append(GuidedPolicyBehaviorRecord.from_mapping(record.to_mapping()))

    first = records[0]
    if first.contract_id != expected_contract_id:
        raise ValueError(
            "joint replay contract does not match expected contract: "
            f"record={first.contract_id}, expected={expected_contract_id}"
        )
    if first.snapshot_id != expected_snapshot_id:
        raise ValueError(
            "joint replay snapshot does not match expected snapshot: "
            f"record={first.snapshot_id}, expected={expected_snapshot_id}"
        )

    for index, record in enumerate(records[1:], start=1):
        if record.contract_id != first.contract_id:
            raise ValueError(
                "joint replay batch mixes behavior contracts: "
                f"index={index}, first={first.contract_id}, actual={record.contract_id}"
            )
        if record.snapshot_id != first.snapshot_id:
            raise ValueError(
                "joint replay batch mixes frozen-Q snapshots: "
                f"index={index}, first={first.snapshot_id}, actual={record.snapshot_id}"
            )
        if (
            record.action_space != first.action_space
            or record.action_space_names != first.action_space_names
            or record.action_token_ids != first.action_token_ids
        ):
            raise ValueError(f"joint replay batch mixes action tables at index {index}")

    expected_shape = (len(records), len(first.action_space_names))
    if tuple(current_prior_logits.shape) != expected_shape:
        raise ValueError(
            "joint replay current_prior_logits shape does not match behavior batch: "
            f"actual={tuple(current_prior_logits.shape)}, expected={expected_shape}"
        )

    persisted_frozen_q = torch.tensor(
        [record.frozen_all_action_q for record in records],
        dtype=current_prior_logits.dtype,
        device=current_prior_logits.device,
    )
    replay = frozen_q_guided_log_probs(
        current_prior_logits,
        persisted_frozen_q,
        first.policy_config,
    )
    guided_action_ids = torch.tensor(
        [record.guided_action_id for record in records],
        dtype=torch.long,
        device=current_prior_logits.device,
    )
    row_ids = torch.arange(len(records), device=current_prior_logits.device)
    current_guided_log_probs = replay["guided_log_probs"][
        row_ids,
        guided_action_ids,
    ]
    behavior_guided_log_probs = torch.tensor(
        [record.behavior_guided_logprob for record in records],
        dtype=current_prior_logits.dtype,
        device=current_prior_logits.device,
    )
    if not torch.isfinite(behavior_guided_log_probs).all():
        raise ValueError("joint replay behavior guided log-probs must be finite")

    return {
        "current_guided_log_probs": current_guided_log_probs,
        "behavior_guided_log_probs": behavior_guided_log_probs,
        "all_current_guided_log_probs": replay["guided_log_probs"],
        "current_prior_log_probs": replay["prior_log_probs"],
        "guided_action_ids": guided_action_ids,
        "contract_id": first.contract_id,
        "snapshot_id": first.snapshot_id,
        "behavior_record_ids": tuple(record.record_id() for record in records),
    }


__all__ = ["replay_guided_behavior_log_probs"]
