"""Human-approved, experiment-bound non-production integration gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

JOINT_INTEGRATION_GATE_IMPLEMENTATION = "id171_dp8_resume_smoke_v1"
K4_ID179_INTEGRATION_GATE_IMPLEMENTATION = (
    "id179_k4_single_update_restore_gate_v1"
)
K4_ID180_INTEGRATION_GATE_IMPLEMENTATION = (
    "id180_k4_single_update_restore_gate_v1"
)


@dataclass(frozen=True)
class JointIntegrationGate:
    implementation: str
    experiment_id: int
    phase: str

    def __post_init__(self) -> None:
        contracts = {
            JOINT_INTEGRATION_GATE_IMPLEMENTATION: (
                171,
                {"update_1", "resume_update_2"},
            ),
            K4_ID179_INTEGRATION_GATE_IMPLEMENTATION: (
                179,
                {"update_1", "restore_only"},
            ),
            K4_ID180_INTEGRATION_GATE_IMPLEMENTATION: (
                180,
                {"update_1", "restore_only"},
            ),
        }
        if self.implementation not in contracts:
            raise ValueError("unsupported joint integration gate implementation")
        experiment_id, phases = contracts[self.implementation]
        if self.experiment_id != experiment_id:
            raise ValueError(
                f"joint integration gate is restricted to experiment {experiment_id}"
            )
        if self.phase not in phases:
            raise ValueError("joint integration gate phase is invalid")

    @property
    def expected_total_training_steps(self) -> int:
        if self.implementation in {
            K4_ID179_INTEGRATION_GATE_IMPLEMENTATION,
            K4_ID180_INTEGRATION_GATE_IMPLEMENTATION,
        }:
            return 1
        return 1 if self.phase == "update_1" else 2

    @property
    def expected_resume_mode(self) -> str:
        return "disable" if self.phase == "update_1" else "auto"


def parse_joint_integration_gate(
    raw: Mapping[str, Any],
) -> JointIntegrationGate | None:
    if not isinstance(raw, Mapping):
        raise ValueError("joint_integration_gate must be a mapping")
    fields = {"enabled", "implementation", "experiment_id", "phase"}
    missing = {"enabled"} - set(raw)
    unexpected = set(raw) - fields
    if missing or unexpected:
        raise ValueError(
            "joint_integration_gate fields are invalid: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    if not isinstance(raw["enabled"], bool):
        raise ValueError("joint_integration_gate.enabled must be explicit bool")
    if not raw["enabled"]:
        populated = {
            field for field in fields - {"enabled"} if raw.get(field) is not None
        }
        if populated:
            raise ValueError(
                "disabled joint_integration_gate has populated fields: "
                f"{sorted(populated)}"
            )
        return None
    required = fields - {"enabled"}
    missing = required - set(raw)
    if missing:
        raise ValueError(
            f"joint_integration_gate is missing fields: {sorted(missing)}"
        )
    return JointIntegrationGate(
        implementation=raw["implementation"],
        experiment_id=raw["experiment_id"],
        phase=raw["phase"],
    )


__all__ = [
    "JOINT_INTEGRATION_GATE_IMPLEMENTATION",
    "K4_ID179_INTEGRATION_GATE_IMPLEMENTATION",
    "K4_ID180_INTEGRATION_GATE_IMPLEMENTATION",
    "JointIntegrationGate",
    "parse_joint_integration_gate",
]
