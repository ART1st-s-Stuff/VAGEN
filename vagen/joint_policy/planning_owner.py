"""CPU lifecycle owner for a TP-rank-zero frozen K4 planner."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ray

from nimloth.training.rl.joint_frozen_q_owner import FrozenQBatchPin
from nimloth.training.rl.joint_planning_scoring import (
    FrozenK4PlanningScoringRecord,
    k4_scoring_record_from_policy_state,
)

from .planning_contract import K4MCTSGuidedPolicyConfig

FROZEN_K4_PLANNER_TRANSPORT_SCHEMA = "vagen_frozen_k4_planner_transport_v1"
FROZEN_K4_OWNER_SCORE_RESULT_SCHEMA = "vagen_frozen_k4_owner_score_result_v1"
FROZEN_K4_OWNER_CHECKPOINT_SCHEMA = "vagen_frozen_k4_owner_checkpoint_v1"


@dataclass(frozen=True)
class FrozenK4PlannerTransport:
    """Small descriptor for a fingerprinted snapshot file on shared storage."""

    schema: str
    transport_path: str
    snapshot_id: str
    snapshot_source_step: int
    contract_id: str
    score_dtype: str
    planning_horizon: int
    mcts_num_simulations: int
    mcts_exploration_constant: float

    def __post_init__(self) -> None:
        if self.schema != FROZEN_K4_PLANNER_TRANSPORT_SCHEMA:
            raise ValueError(
                f"unsupported K4 planner transport schema: {self.schema!r}"
            )
        path = Path(self.transport_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"missing K4 planner transport: {path}")
        object.__setattr__(self, "transport_path", str(path))
        for field in ("snapshot_id", "contract_id"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise ValueError(f"K4 planner transport {field} must be non-empty")
        if (
            isinstance(self.snapshot_source_step, bool)
            or not isinstance(self.snapshot_source_step, int)
            or self.snapshot_source_step < 0
        ):
            raise ValueError("K4 planner source step must be a non-negative int")
        if self.score_dtype not in {"float32", "bfloat16", "float64"}:
            raise ValueError("K4 planner transport score_dtype is unsupported")
        config = K4MCTSGuidedPolicyConfig.from_mapping(
            {
                "implementation": "k4_mcts_guided_v1",
                "alpha": 1.0,
                "beta": 0.0,
                "prior_temperature": 1.0,
                "backprop_to_llm": True,
                "score_dtype": self.score_dtype,
                "planning_horizon": self.planning_horizon,
                "mcts_num_simulations": self.mcts_num_simulations,
                "mcts_exploration_constant": self.mcts_exploration_constant,
            }
        )
        object.__setattr__(self, "planning_horizon", config.planning_horizon)
        object.__setattr__(
            self,
            "mcts_num_simulations",
            config.mcts_num_simulations,
        )
        object.__setattr__(
            self,
            "mcts_exploration_constant",
            config.mcts_exploration_constant,
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FrozenK4PlannerTransport":
        values = _exact_mapping(
            raw,
            frozenset(cls.__dataclass_fields__),
            "K4 planner transport",
        )
        return cls(**values)

    def to_mapping(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in self.__dataclass_fields__
        }


@ray.remote(num_cpus=1, num_gpus=0, max_restarts=0, max_task_retries=0)
class FrozenK4PlanningActor:
    """Own pins/CAS while all expensive planning runs in the vLLM TP rank zero."""

    def __init__(
        self,
        initial_transport: Mapping[str, Any],
        policy_config: Mapping[str, Any],
        server_handles: list[Any],
        *,
        activation_version: int = 0,
    ) -> None:
        resources = ray.get_runtime_context().get_assigned_resources()
        if float(resources.get("CPU", 0.0)) != 1.0 or float(resources.get("GPU", 0.0)) != 0.0:
            raise RuntimeError("K4 planning actor requires exactly one CPU and no GPU")
        self._policy = K4MCTSGuidedPolicyConfig.from_mapping(policy_config)
        self._active = FrozenK4PlannerTransport.from_mapping(initial_transport)
        self._activation_version = _nonnegative_int(
            activation_version,
            "activation_version",
        )
        if len(server_handles) != 1:
            raise ValueError("K4 TP8/DP1 planning requires exactly one rollout server")
        self._server = server_handles[0]
        self._staged: FrozenK4PlannerTransport | None = None
        self._pins: dict[str, FrozenQBatchPin] = {}
        self._validate_transport(self._active)
        self._install(self._active)

    def _validate_transport(self, transport: FrozenK4PlannerTransport) -> None:
        if (
            transport.score_dtype != self._policy.score_dtype
            or transport.planning_horizon != self._policy.planning_horizon
            or transport.mcts_num_simulations != self._policy.mcts_num_simulations
            or transport.mcts_exploration_constant
            != self._policy.mcts_exploration_constant
        ):
            raise ValueError("K4 planner transport does not match policy search contract")

    def _install(self, transport: FrozenK4PlannerTransport) -> None:
        status = ray.get(
            self._server.install_frozen_k4_planner.remote(
                {
                    "transport_path": transport.transport_path,
                    "expected_snapshot_id": transport.snapshot_id,
                    "expected_source_step": transport.snapshot_source_step,
                    "expected_contract_id": transport.contract_id,
                    "expected_activation_version": self._activation_version,
                }
            )
        )
        expected = {
            "snapshot_id": transport.snapshot_id,
            "source_step": transport.snapshot_source_step,
            "contract_id": transport.contract_id,
            "activation_version": self._activation_version,
            "transport_path": transport.transport_path,
        }
        if status != expected:
            raise RuntimeError("K4 rollout server installed a different planner identity")

    def status(self) -> dict[str, Any]:
        return {
            "active_snapshot_id": self._active.snapshot_id,
            "active_source_step": self._active.snapshot_source_step,
            "contract_id": self._active.contract_id,
            "score_dtype": self._active.score_dtype,
            "activation_version": self._activation_version,
            "staged_snapshot_id": (
                None if self._staged is None else self._staged.snapshot_id
            ),
            "open_batch_count": len(self._pins),
            "planner_placement": "vllm_tp_rank_zero",
        }

    def pin_batch(self, request: Mapping[str, Any]) -> dict[str, Any]:
        values = _exact_mapping(
            request,
            {
                "batch_id",
                "policy_step",
                "expected_snapshot_id",
                "expected_activation_version",
            },
            "K4 pin request",
        )
        if (
            values["expected_snapshot_id"] != self._active.snapshot_id
            or values["expected_activation_version"] != self._activation_version
        ):
            raise ValueError("K4 planner pin compare-and-swap mismatch")
        pin = FrozenQBatchPin(
            schema="nimloth_frozen_q_batch_pin_v1",
            batch_id=values["batch_id"],
            policy_step=values["policy_step"],
            snapshot_id=self._active.snapshot_id,
            snapshot_source_step=self._active.snapshot_source_step,
            contract_id=self._active.contract_id,
            activation_version=self._activation_version,
        )
        existing = self._pins.get(pin.batch_id)
        if existing is not None and existing != pin:
            raise ValueError("K4 planner batch id is already pinned differently")
        self._pins[pin.batch_id] = pin
        return pin.to_mapping()

    def unpin_batch(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        pin = FrozenQBatchPin.from_mapping(raw)
        if self._pins.get(pin.batch_id) != pin:
            raise ValueError("K4 planner unpin does not match an open pin")
        del self._pins[pin.batch_id]
        return self.status()

    def score(self, request: Mapping[str, Any]) -> dict[str, Any]:
        fields = {
            "schema",
            "batch_pin",
            "policy_state",
            "expected_request_id",
            "expected_generation_id",
            "expected_latent_token_ids",
            "expected_action_start_token_id",
            "expected_action_token_ids",
            "expected_contract_id",
        }
        values = _exact_mapping(request, fields, "K4 score request")
        if values["schema"] != "nimloth_frozen_q_owner_score_request_v1":
            raise ValueError("K4 score request schema is invalid")
        pin = FrozenQBatchPin.from_mapping(values["batch_pin"])
        if self._pins.get(pin.batch_id) != pin:
            raise ValueError("K4 score request does not match an open pin")
        if (
            pin.snapshot_id != self._active.snapshot_id
            or pin.snapshot_source_step != self._active.snapshot_source_step
            or pin.contract_id != self._active.contract_id
            or values["expected_contract_id"] != self._active.contract_id
        ):
            raise ValueError("K4 score request identity does not match active planner")
        record = k4_scoring_record_from_policy_state(
            values["policy_state"],
            expected_request_id=values["expected_request_id"],
            expected_generation_id=values["expected_generation_id"],
            expected_latent_token_ids=values["expected_latent_token_ids"],
            expected_action_start_token_id=values["expected_action_start_token_id"],
            expected_action_token_ids=values["expected_action_token_ids"],
            expected_snapshot_id=self._active.snapshot_id,
            expected_snapshot_source_step=self._active.snapshot_source_step,
            expected_contract_id=self._active.contract_id,
            expected_activation_version=self._activation_version,
            expected_score_dtype=self._active.score_dtype,
            expected_planning_horizon=self._policy.planning_horizon,
            expected_mcts_num_simulations=self._policy.mcts_num_simulations,
            expected_mcts_exploration_constant=(
                self._policy.mcts_exploration_constant
            ),
        )
        return {
            "schema": FROZEN_K4_OWNER_SCORE_RESULT_SCHEMA,
            "batch_pin": pin.to_mapping(),
            "scoring_record": record.to_mapping(),
        }

    def stage_snapshot(self, request: Mapping[str, Any]) -> dict[str, Any]:
        values = _exact_mapping(
            request,
            {
                "new_snapshot_state",
                "expected_active_snapshot_id",
                "expected_activation_version",
            },
            "K4 stage request",
        )
        self._validate_cas(
            values["expected_active_snapshot_id"],
            values["expected_activation_version"],
        )
        candidate = FrozenK4PlannerTransport.from_mapping(
            values["new_snapshot_state"]
        )
        self._validate_transport(candidate)
        if (
            candidate.snapshot_source_step <= self._active.snapshot_source_step
            or candidate.contract_id != self._active.contract_id
        ):
            raise ValueError("staged K4 planner lineage is invalid")
        if self._staged is not None and self._staged != candidate:
            raise ValueError("a different K4 planner snapshot is already staged")
        self._staged = candidate
        return self.status()

    def activate_staged(self, request: Mapping[str, Any]) -> dict[str, Any]:
        values = _exact_mapping(
            request,
            {
                "staged_snapshot_id",
                "expected_active_snapshot_id",
                "expected_activation_version",
            },
            "K4 activate request",
        )
        self._validate_cas(
            values["expected_active_snapshot_id"],
            values["expected_activation_version"],
        )
        if self._pins:
            raise ValueError("cannot activate K4 planner with open batch pins")
        if self._staged is None or self._staged.snapshot_id != values["staged_snapshot_id"]:
            raise ValueError("K4 staged planner identity mismatch")
        candidate = self._staged
        self._activation_version += 1
        try:
            self._install(candidate)
        except BaseException:
            self._activation_version -= 1
            raise
        self._active = candidate
        self._staged = None
        return self.status()

    def checkpoint_state(self) -> dict[str, Any]:
        if self._pins or self._staged is not None:
            raise ValueError("cannot checkpoint K4 planner with pins or staged state")
        return {
            "schema": FROZEN_K4_OWNER_CHECKPOINT_SCHEMA,
            "activation_version": self._activation_version,
            "active_snapshot_state": self._active.to_mapping(),
        }

    def restore_checkpoint_state(self, raw: Mapping[str, Any]) -> dict[str, Any]:
        values = _exact_mapping(
            raw,
            {"schema", "activation_version", "active_snapshot_state"},
            "K4 owner checkpoint",
        )
        if values["schema"] != FROZEN_K4_OWNER_CHECKPOINT_SCHEMA:
            raise ValueError("K4 owner checkpoint schema is invalid")
        if self._pins or self._staged is not None:
            raise ValueError("cannot restore K4 owner with pins or staged state")
        candidate = FrozenK4PlannerTransport.from_mapping(
            values["active_snapshot_state"]
        )
        self._validate_transport(candidate)
        old_version = self._activation_version
        self._activation_version = _nonnegative_int(
            values["activation_version"],
            "activation_version",
        )
        try:
            self._install(candidate)
        except BaseException:
            self._activation_version = old_version
            raise
        self._active = candidate
        return self.status()

    def _validate_cas(self, snapshot_id: Any, activation_version: Any) -> None:
        if (
            snapshot_id != self._active.snapshot_id
            or activation_version != self._activation_version
        ):
            raise ValueError("K4 planner compare-and-swap mismatch")


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"K4 planning owner {field} must be non-negative int")
    return value


def _exact_mapping(
    raw: Mapping[str, Any],
    fields: set[str] | frozenset[str],
    context: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a mapping")
    missing = set(fields) - set(raw)
    if missing:
        raise ValueError(f"{context} is missing fields: {sorted(missing)}")
    unexpected = set(raw) - set(fields)
    if unexpected:
        raise ValueError(f"{context} has unexpected fields: {sorted(unexpected)}")
    return {field: raw[field] for field in fields}


__all__ = [
    "FROZEN_K4_OWNER_CHECKPOINT_SCHEMA",
    "FROZEN_K4_OWNER_SCORE_RESULT_SCHEMA",
    "FROZEN_K4_PLANNER_TRANSPORT_SCHEMA",
    "FrozenK4PlannerTransport",
    "FrozenK4PlanningActor",
]
