"""Dedicated one-CPU Ray wrapper for Nimloth's frozen-Q snapshot owner."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import ray
import torch

from nimloth.training.rl.joint_frozen_q_owner import FrozenQSnapshotOwner


@ray.remote(
    num_cpus=1,
    num_gpus=0,
    max_restarts=0,
    max_task_retries=0,
)
class FrozenQScoringActor:
    """Serialize all snapshot scoring and lifecycle mutations on one CPU actor."""

    def __init__(
        self,
        initial_snapshot_state: Mapping[str, Any],
        *,
        activation_version: int = 0,
    ) -> None:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
            raise RuntimeError(
                "frozen Q scoring actor must limit PyTorch intra-op and inter-op threads to one"
            )
        resources = ray.get_runtime_context().get_assigned_resources()
        if float(resources.get("CPU", 0.0)) != 1.0:
            raise RuntimeError(
                "frozen Q scoring actor requires exactly one assigned CPU"
            )
        if float(resources.get("GPU", 0.0)) != 0.0:
            raise RuntimeError("frozen Q scoring actor must not receive a GPU")
        self._owner = FrozenQSnapshotOwner(
            initial_snapshot_state=initial_snapshot_state,
            activation_version=activation_version,
        )

    def status(self) -> dict[str, Any]:
        return {
            **self._owner.status(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
        }

    def pin_batch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self._owner.pin_batch(
            **_mapping(
                request,
                "pin_batch",
                {
                    "batch_id",
                    "policy_step",
                    "expected_snapshot_id",
                    "expected_activation_version",
                },
            )
        )

    def unpin_batch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self._owner.unpin_batch(_mapping(request, "unpin_batch"))

    def score(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self._owner.score(_mapping(request, "score"))

    def stage_snapshot(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self._owner.stage_snapshot(
            **_mapping(
                request,
                "stage_snapshot",
                {
                    "new_snapshot_state",
                    "expected_active_snapshot_id",
                    "expected_activation_version",
                },
            )
        )

    def activate_staged(self, request: Mapping[str, Any]) -> dict[str, Any]:
        return self._owner.activate_staged(
            **_mapping(
                request,
                "activate_staged",
                {
                    "staged_snapshot_id",
                    "expected_active_snapshot_id",
                    "expected_activation_version",
                },
            )
        )

    def checkpoint_state(self) -> dict[str, Any]:
        return self._owner.checkpoint_state()


def _mapping(
    value: Mapping[str, Any],
    method: str,
    fields: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"frozen Q actor {method} request must be a mapping")
    result = dict(value)
    if fields is not None:
        missing = fields - set(result)
        if missing:
            raise ValueError(
                f"frozen Q actor {method} request is missing fields: {sorted(missing)}"
            )
        unexpected = set(result) - fields
        if unexpected:
            raise ValueError(
                "frozen Q actor "
                f"{method} request has unexpected fields: {sorted(unexpected)}"
            )
    return result


__all__ = ["FrozenQScoringActor"]
