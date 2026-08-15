"""Explicit online world-model contract for K4 joint training."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

_K4_WM_IMPLEMENTATION = "k4_world_model_update_v1"
_K4_WM_FIELDS = frozenset(
    {
        "implementation",
        "planning_checkpoint",
        "snapshot_transport_root",
        "prediction_horizon",
        "minimum_window_depth",
        "maximum_window_depth",
        "state_mse_weight",
        "dino_grid_weight",
        "sigreg_weight",
        "sigreg_knots",
        "sigreg_num_proj",
        "dino_identity",
        "selected_action_huber_delta",
        "grad_clip",
        "optimizer",
    }
)
_DINO_FIELDS = frozenset(
    {"source", "revision", "processor_fingerprint", "hidden_size", "grid_size"}
)
_EXPECTED_DINO = {
    "source": "facebook/dinov2-large",
    "revision": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
    "processor_fingerprint": "7d65a7de8788e87d",
    "hidden_size": 1024,
    "grid_size": 4,
}


@dataclass(frozen=True)
class K4PlanningOptimizerConfig:
    """One AdamW with explicit projector/predictor/ValueHead groups."""

    name: str
    projector_lr: float
    predictor_lr: float
    value_head_lr: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "K4PlanningOptimizerConfig":
        fields = frozenset(cls.__dataclass_fields__)
        values = _exact_mapping(raw, fields, "K4 planning optimizer")
        if values["name"] != "adamw":
            raise ValueError("K4 planning optimizer supports only adamw")
        beta_values = _plain_sequence(values["betas"], "K4 planning optimizer betas")
        if len(beta_values) != 2:
            raise ValueError("K4 planning optimizer betas must contain two values")
        betas = tuple(_finite_float(value, "K4 planning optimizer beta") for value in beta_values)
        if any(value < 0.0 or value >= 1.0 for value in betas):
            raise ValueError("K4 planning optimizer betas must be in [0, 1)")
        return cls(
            name="adamw",
            projector_lr=_positive_float(values["projector_lr"], "projector lr"),
            predictor_lr=_positive_float(values["predictor_lr"], "predictor lr"),
            value_head_lr=_positive_float(values["value_head_lr"], "value head lr"),
            betas=(betas[0], betas[1]),
            eps=_positive_float(values["eps"], "planning optimizer eps"),
            weight_decay=_nonnegative_float(
                values["weight_decay"],
                "planning optimizer weight decay",
            ),
        )


@dataclass(frozen=True)
class K4WorldModelTrainingConfig:
    """No-default 1--4-step WM/DINO/SIGReg and critic update contract."""

    implementation: str
    planning_checkpoint: str
    snapshot_transport_root: str
    prediction_horizon: int
    minimum_window_depth: int
    maximum_window_depth: int
    state_mse_weight: float
    dino_grid_weight: float
    sigreg_weight: float
    sigreg_knots: int
    sigreg_num_proj: int
    dino_identity: dict[str, Any]
    selected_action_huber_delta: float
    grad_clip: float
    optimizer: K4PlanningOptimizerConfig

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "K4WorldModelTrainingConfig":
        values = _exact_mapping(raw, _K4_WM_FIELDS, "K4 world-model training config")
        if values["implementation"] != _K4_WM_IMPLEMENTATION:
            raise ValueError("unsupported K4 world-model training implementation")
        checkpoint = _nonempty_string(values["planning_checkpoint"], "planning_checkpoint")
        transport = _nonempty_string(
            values["snapshot_transport_root"],
            "snapshot_transport_root",
        )
        horizon = _positive_int(values["prediction_horizon"], "prediction_horizon")
        minimum = _positive_int(values["minimum_window_depth"], "minimum_window_depth")
        maximum = _positive_int(values["maximum_window_depth"], "maximum_window_depth")
        if horizon != 4 or minimum != 1 or maximum != 4:
            raise ValueError("K4 world-model windows must be all depths 1 through 4")
        dino = _exact_mapping(values["dino_identity"], _DINO_FIELDS, "DINO identity")
        if dino != _EXPECTED_DINO:
            raise ValueError("K4 world-model DINO identity is not the ID74 teacher")
        knots = _positive_int(values["sigreg_knots"], "sigreg_knots")
        projections = _positive_int(values["sigreg_num_proj"], "sigreg_num_proj")
        if knots != 17 or projections != 1024:
            raise ValueError("K4 world-model SIGReg shape must reuse ID74")
        return cls(
            implementation=_K4_WM_IMPLEMENTATION,
            planning_checkpoint=checkpoint,
            snapshot_transport_root=transport,
            prediction_horizon=horizon,
            minimum_window_depth=minimum,
            maximum_window_depth=maximum,
            state_mse_weight=_positive_float(values["state_mse_weight"], "state MSE weight"),
            dino_grid_weight=_positive_float(values["dino_grid_weight"], "DINO-grid weight"),
            sigreg_weight=_positive_float(values["sigreg_weight"], "SIGReg weight"),
            sigreg_knots=knots,
            sigreg_num_proj=projections,
            dino_identity=dict(dino),
            selected_action_huber_delta=_positive_float(
                values["selected_action_huber_delta"],
                "selected-action Huber delta",
            ),
            grad_clip=_positive_float(values["grad_clip"], "planning grad clip"),
            optimizer=K4PlanningOptimizerConfig.from_mapping(values["optimizer"]),
        )


def parse_k4_world_model_training_section(
    raw: Mapping[str, Any],
) -> K4WorldModelTrainingConfig | None:
    """Parse explicit opt-in without filling any joint training value."""

    if not isinstance(raw, Mapping):
        raise ValueError("k4_world_model_training must be a mapping")
    if "enabled" not in raw or not isinstance(raw["enabled"], bool):
        raise ValueError("k4_world_model_training.enabled must be explicit bool")
    unexpected = set(raw) - (_K4_WM_FIELDS | {"enabled"})
    if unexpected:
        raise ValueError(
            "k4_world_model_training has unexpected fields: "
            f"{sorted(unexpected)}"
        )
    if not raw["enabled"]:
        populated = {field for field in _K4_WM_FIELDS if raw.get(field) is not None}
        if populated:
            raise ValueError(
                "disabled k4_world_model_training has populated fields: "
                f"{sorted(populated)}"
            )
        return None
    missing = _K4_WM_FIELDS - set(raw)
    if missing:
        raise ValueError(
            "k4_world_model_training is missing fields: "
            f"{sorted(missing)}"
        )
    return K4WorldModelTrainingConfig.from_mapping(
        {field: raw[field] for field in _K4_WM_FIELDS}
    )


def k4_world_model_training_contract_id(
    config: K4WorldModelTrainingConfig,
) -> str:
    if not isinstance(config, K4WorldModelTrainingConfig):
        raise TypeError("K4 WM contract ID requires K4WorldModelTrainingConfig")
    payload = json.dumps(
        asdict(config),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _exact_mapping(raw: Any, fields: frozenset[str], context: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{context} must be a mapping")
    missing = set(fields) - set(raw)
    unexpected = set(raw) - set(fields)
    if missing or unexpected:
        raise ValueError(
            f"{context} fields are invalid: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return {field: raw[field] for field in fields}


def _plain_sequence(value: Any, field: str) -> list[Any]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f"{field} must be a plain sequence")
    return list(value)


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return 0.0 if result == 0.0 else result


def _positive_float(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative_float(value: Any, field: str) -> float:
    result = _finite_float(value, field)
    if result < 0.0:
        raise ValueError(f"{field} must be non-negative")
    return result


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive int")
    return value


def _nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


__all__ = [
    "K4PlanningOptimizerConfig",
    "K4WorldModelTrainingConfig",
    "k4_world_model_training_contract_id",
    "parse_k4_world_model_training_section",
]
