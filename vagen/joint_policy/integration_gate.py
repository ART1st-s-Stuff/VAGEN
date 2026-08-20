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
K4_ID181_INTEGRATION_GATE_IMPLEMENTATION = (
    "id181_k4_single_update_restore_gate_v1"
)
K4_ID182_INTEGRATION_GATE_IMPLEMENTATION = (
    "id182_k4_single_update_restore_gate_v1"
)
K4_ID183_CANARY_GATE_IMPLEMENTATION = "id183_k4_10update_canary_v1"
K4_ID184_CONTINUE_GATE_IMPLEMENTATION = "id184_k4_continue_to20_v1"
K4_ID185_FULL_EVAL_GATE_IMPLEMENTATION = "id185_k4_full_eval_test300_v1"
K4_ID186_CONTINUE_GATE_IMPLEMENTATION = "id186_k4_continue_to40_v1"
K4_ID187_SOURCE20_BROWSER_GATE_IMPLEMENTATION = "id187_k4_source20_browser_v1"
K4_ID188_STEP0_BROWSER_GATE_IMPLEMENTATION = "id188_k4_step0_browser_v1"


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
            K4_ID181_INTEGRATION_GATE_IMPLEMENTATION: (
                181,
                {"update_1", "restore_only"},
            ),
            K4_ID182_INTEGRATION_GATE_IMPLEMENTATION: (
                182,
                {"update_1", "restore_only"},
            ),
            K4_ID183_CANARY_GATE_IMPLEMENTATION: (
                183,
                {"train_to_5", "resume_to_10"},
            ),
            K4_ID184_CONTINUE_GATE_IMPLEMENTATION: (
                184,
                {"resume_10_to_20"},
            ),
            K4_ID185_FULL_EVAL_GATE_IMPLEMENTATION: (
                185,
                {"full_eval_test300", "visualize_one"},
            ),
            K4_ID186_CONTINUE_GATE_IMPLEMENTATION: (
                186,
                {"resume_20_to_30", "resume_30_to_40"},
            ),
            K4_ID187_SOURCE20_BROWSER_GATE_IMPLEMENTATION: (
                187,
                {"source20_visualize_one"},
            ),
            K4_ID188_STEP0_BROWSER_GATE_IMPLEMENTATION: (
                188,
                {"step0_visualize_one"},
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
        if self.implementation == K4_ID183_CANARY_GATE_IMPLEMENTATION:
            return 5 if self.phase == "train_to_5" else 10
        if self.implementation in {
            K4_ID184_CONTINUE_GATE_IMPLEMENTATION,
            K4_ID185_FULL_EVAL_GATE_IMPLEMENTATION,
        }:
            return 20
        if self.implementation == K4_ID186_CONTINUE_GATE_IMPLEMENTATION:
            return 30 if self.phase == "resume_20_to_30" else 40
        if self.implementation == K4_ID187_SOURCE20_BROWSER_GATE_IMPLEMENTATION:
            return 20
        if self.implementation == K4_ID188_STEP0_BROWSER_GATE_IMPLEMENTATION:
            return 1
        if self.implementation in {
            K4_ID179_INTEGRATION_GATE_IMPLEMENTATION,
            K4_ID180_INTEGRATION_GATE_IMPLEMENTATION,
            K4_ID181_INTEGRATION_GATE_IMPLEMENTATION,
            K4_ID182_INTEGRATION_GATE_IMPLEMENTATION,
        }:
            return 1
        return 1 if self.phase == "update_1" else 2

    @property
    def expected_resume_mode(self) -> str:
        if self.implementation in {
            K4_ID184_CONTINUE_GATE_IMPLEMENTATION,
            K4_ID185_FULL_EVAL_GATE_IMPLEMENTATION,
            K4_ID186_CONTINUE_GATE_IMPLEMENTATION,
            K4_ID187_SOURCE20_BROWSER_GATE_IMPLEMENTATION,
        }:
            return "resume_path"
        return (
            "disable"
            if self.phase in {"update_1", "train_to_5", "step0_visualize_one"}
            else "auto"
        )


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
    "K4_ID181_INTEGRATION_GATE_IMPLEMENTATION",
    "K4_ID182_INTEGRATION_GATE_IMPLEMENTATION",
    "K4_ID183_CANARY_GATE_IMPLEMENTATION",
    "K4_ID184_CONTINUE_GATE_IMPLEMENTATION",
    "K4_ID185_FULL_EVAL_GATE_IMPLEMENTATION",
    "K4_ID186_CONTINUE_GATE_IMPLEMENTATION",
    "K4_ID187_SOURCE20_BROWSER_GATE_IMPLEMENTATION",
    "K4_ID188_STEP0_BROWSER_GATE_IMPLEMENTATION",
    "JointIntegrationGate",
    "parse_joint_integration_gate",
]
