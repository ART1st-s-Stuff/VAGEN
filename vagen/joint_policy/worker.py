"""VAGEN worker RPCs for the custom guided actor and replicated critic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker


class JointAsyncActorRolloutRefWorker(AsyncActorRolloutRefWorker):
    """Expose rank-consistency evidence after a complete custom actor update."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def export_joint_critic_snapshot(
        self,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise ValueError("joint critic snapshot export request must be a mapping")
        required = {"source_step", "contract_id", "score_dtype"}
        missing = required - set(request)
        if missing:
            raise ValueError(
                f"joint critic snapshot export is missing fields: {sorted(missing)}"
            )
        unexpected = set(request) - required
        if unexpected:
            raise ValueError(
                "joint critic snapshot export has unexpected fields: "
                f"{sorted(unexpected)}"
            )
        if not hasattr(self.actor, "export_joint_critic_snapshot"):
            raise RuntimeError("worker actor does not implement joint critic export")
        return self.actor.export_joint_critic_snapshot(
            source_step=request["source_step"],
            contract_id=request["contract_id"],
            score_dtype=request["score_dtype"],
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def export_joint_checkpoint(
        self,
        request: Mapping[str, Any],
    ) -> dict[str, Any]:
        values = _exact_request(
            request,
            "export_joint_checkpoint",
            {"source_step", "contract_id", "score_dtype"},
        )
        if not hasattr(self.actor, "export_joint_checkpoint"):
            raise RuntimeError("worker actor does not implement joint checkpoint export")
        return self.actor.export_joint_checkpoint(**values)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_joint_checkpoint(
        self,
        checkpoint: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(checkpoint, Mapping):
            raise ValueError("joint worker checkpoint must be a mapping")
        if not hasattr(self.actor, "load_joint_checkpoint"):
            raise RuntimeError("worker actor does not implement joint checkpoint load")
        return self.actor.load_joint_checkpoint(checkpoint)


def _exact_request(
    raw: Mapping[str, Any],
    method: str,
    fields: set[str],
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"joint worker {method} request must be a mapping")
    missing = fields - set(raw)
    unexpected = set(raw) - fields
    if missing or unexpected:
        raise ValueError(
            f"joint worker {method} request fields are invalid: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return dict(raw)


__all__ = ["JointAsyncActorRolloutRefWorker"]
