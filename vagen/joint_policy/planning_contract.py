"""Dependency-light Scheme-B contract for K4 MCTS guidance."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

K4_MCTS_GUIDED_BEHAVIOR_SCHEMA = "vagen_k4_mcts_guided_behavior_v1"
_K4_IMPLEMENTATION = "k4_mcts_guided_v1"
_K4_CONFIG_FIELDS = frozenset(
    {
        "implementation",
        "alpha",
        "beta",
        "prior_temperature",
        "backprop_to_llm",
        "score_dtype",
        "planning_horizon",
        "mcts_num_simulations",
        "mcts_exploration_constant",
    }
)


@dataclass(frozen=True)
class K4MCTSGuidedPolicyConfig:
    """Explicit policy and search fields that define K4 behavior identity."""

    implementation: str
    alpha: float
    beta: float
    prior_temperature: float
    backprop_to_llm: bool
    score_dtype: str
    planning_horizon: int
    mcts_num_simulations: int
    mcts_exploration_constant: float

    def __post_init__(self) -> None:
        values = type(self)._validated_fields(asdict(self))
        for field, value in values.items():
            object.__setattr__(self, field, value)

    @classmethod
    def _validated_fields(cls, raw: Mapping[str, Any]) -> dict[str, Any]:
        values = _exact_mapping(raw, _K4_CONFIG_FIELDS, "K4 joint policy config")
        if values["implementation"] != _K4_IMPLEMENTATION:
            raise ValueError(
                f"unsupported K4 joint policy implementation: {values['implementation']!r}"
            )
        alpha = _finite_float(values["alpha"], "alpha")
        beta = _finite_float(values["beta"], "beta")
        prior_temperature = _finite_float(
            values["prior_temperature"],
            "prior_temperature",
        )
        if alpha <= 0.0:
            raise ValueError("K4 joint policy alpha must be positive")
        if beta < 0.0:
            raise ValueError("K4 joint policy beta must be non-negative")
        if prior_temperature <= 0.0:
            raise ValueError(
                "K4 joint policy prior_temperature must be positive"
            )
        if values["backprop_to_llm"] is not True:
            raise ValueError("K4 joint policy backprop_to_llm must be true")
        if values["score_dtype"] not in {"float32", "bfloat16", "float64"}:
            raise ValueError("K4 joint policy score_dtype is unsupported")
        horizon = _positive_int(values["planning_horizon"], "planning_horizon")
        if horizon != 4:
            raise ValueError("K4 joint policy planning_horizon must be exactly 4")
        simulations = _positive_int(
            values["mcts_num_simulations"],
            "mcts_num_simulations",
        )
        exploration = _finite_float(
            values["mcts_exploration_constant"],
            "mcts_exploration_constant",
        )
        if exploration < 0.0:
            raise ValueError(
                "K4 joint policy mcts_exploration_constant must be non-negative"
            )
        return {
            "implementation": _K4_IMPLEMENTATION,
            "alpha": alpha,
            "beta": 0.0 if beta == 0.0 else beta,
            "prior_temperature": prior_temperature,
            "backprop_to_llm": True,
            "score_dtype": values["score_dtype"],
            "planning_horizon": horizon,
            "mcts_num_simulations": simulations,
            "mcts_exploration_constant": exploration,
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "K4MCTSGuidedPolicyConfig":
        return cls(**cls._validated_fields(raw))

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def contract_id(
        self,
        action_space: str,
        action_space_names: Sequence[str],
        action_token_ids: Sequence[int],
    ) -> str:
        table = _action_table(action_space, action_space_names, action_token_ids)
        return _sha256_id({"config": self.to_mapping(), **table})


def parse_k4_mcts_joint_policy_section(
    raw: Mapping[str, Any],
) -> K4MCTSGuidedPolicyConfig | None:
    """Parse the explicit K4 opt-in without synthesizing search defaults."""

    if not isinstance(raw, Mapping):
        raise ValueError("K4 joint_policy section must be a mapping")
    if "enabled" not in raw or not isinstance(raw["enabled"], bool):
        raise ValueError("K4 joint_policy.enabled must be explicit bool")
    allowed = _K4_CONFIG_FIELDS | {"enabled"}
    unexpected = set(raw) - allowed
    if unexpected:
        raise ValueError(
            f"K4 joint_policy section has unexpected fields: {sorted(unexpected)}"
        )
    if not raw["enabled"]:
        return None
    missing = _K4_CONFIG_FIELDS - set(raw)
    if missing:
        raise ValueError(
            f"K4 joint_policy section is missing fields: {sorted(missing)}"
        )
    return K4MCTSGuidedPolicyConfig.from_mapping(
        {field: raw[field] for field in _K4_CONFIG_FIELDS}
    )


def k4_guided_log_probs_reference(
    prior_logits: Sequence[Real],
    planner_root_mean_values: Sequence[Real],
    config: K4MCTSGuidedPolicyConfig,
) -> tuple[list[float], list[float]]:
    """Compute Scheme-B from Qwen prior logits and raw MCTS root means."""

    logits = _finite_vector(prior_logits, "prior_logits")
    planner = _finite_vector(
        planner_root_mean_values,
        "planner_root_mean_values",
    )
    if len(logits) != len(planner):
        raise ValueError("K4 prior logits and planner root values must align")
    scaled_prior = [value / config.prior_temperature for value in logits]
    guided_logits = [
        config.alpha * prior + config.beta * root
        for prior, root in zip(scaled_prior, planner, strict=True)
    ]
    return _log_softmax(scaled_prior), _log_softmax(guided_logits)


@dataclass(frozen=True)
class K4MCTSGuidedBehaviorRecord:
    """Behavior evidence separating direct root Q from MCTS guidance."""

    schema: str
    contract_id: str
    policy_config: K4MCTSGuidedPolicyConfig
    snapshot_id: str
    action_space: str
    action_space_names: tuple[str, ...]
    action_token_ids: tuple[int, ...]
    prior_token_id: int
    prior_action_id: int
    prior_response_idx: int
    behavior_llm_prior_logprob: float
    prior_logits: tuple[float, ...]
    prior_log_probs: tuple[float, ...]
    direct_all_action_q: tuple[float, ...]
    planner_root_mean_values: tuple[float, ...]
    planner_root_visit_counts: tuple[int, ...]
    guided_action_id: int
    behavior_guided_logprob: float

    @classmethod
    def build(
        cls,
        *,
        action_space: str,
        action_space_names: Sequence[str],
        action_token_ids: Sequence[int],
        snapshot_id: str,
        prior_token_id: int,
        prior_action_id: int,
        prior_response_idx: int,
        behavior_llm_prior_logprob: Real,
        prior_logits: Sequence[Real],
        direct_all_action_q: Sequence[Real],
        planner_root_mean_values: Sequence[Real],
        planner_root_visit_counts: Sequence[int],
        guided_action_id: int,
        behavior_guided_logprob: Real,
        config: K4MCTSGuidedPolicyConfig,
    ) -> "K4MCTSGuidedBehaviorRecord":
        table = _action_table(action_space, action_space_names, action_token_ids)
        names = tuple(table["action_space_names"])
        token_ids = tuple(table["action_token_ids"])
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError("K4 guided snapshot_id must be non-empty")
        for field, value in (
            ("prior_token_id", prior_token_id),
            ("prior_action_id", prior_action_id),
            ("prior_response_idx", prior_response_idx),
            ("guided_action_id", guided_action_id),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"K4 guided {field} must be a non-negative int")
        logits = tuple(_finite_vector(prior_logits, "prior_logits"))
        direct_q = tuple(
            _finite_vector(direct_all_action_q, "direct_all_action_q")
        )
        planner = tuple(
            _finite_vector(
                planner_root_mean_values,
                "planner_root_mean_values",
            )
        )
        visits = tuple(
            _root_visits(
                planner_root_visit_counts,
                action_count=len(names),
                expected_total=config.mcts_num_simulations,
            )
        )
        if len(logits) != len(names) or len(direct_q) != len(names) or len(planner) != len(names):
            raise ValueError(
                "K4 guided logits, direct Q, planner values, and actions must align"
            )
        if prior_action_id >= len(names) or guided_action_id >= len(names):
            raise ValueError("K4 guided action id is outside action space")
        if prior_token_id != token_ids[prior_action_id]:
            raise ValueError("K4 guided prior token does not match action table")
        prior_log_probs, guided_log_probs = k4_guided_log_probs_reference(
            logits,
            planner,
            config,
        )
        # This log-prob belongs to the sampled response token distribution, whose
        # CoT temperature/top-p can differ from Scheme-B's prior_temperature.
        llm_logprob = _finite_float(
            behavior_llm_prior_logprob,
            "behavior_llm_prior_logprob",
        )
        guided_logprob = _finite_float(
            behavior_guided_logprob,
            "behavior_guided_logprob",
        )
        expected_guided = guided_log_probs[guided_action_id]
        if not math.isclose(
            guided_logprob,
            expected_guided,
            rel_tol=0.0,
            abs_tol=_logprob_tolerance(config.score_dtype, expected_guided),
        ):
            raise ValueError(
                "K4 behavior guided log-prob does not match planner-guided distribution"
            )
        return cls(
            schema=K4_MCTS_GUIDED_BEHAVIOR_SCHEMA,
            contract_id=config.contract_id(action_space, names, token_ids),
            policy_config=config,
            snapshot_id=snapshot_id,
            action_space=action_space,
            action_space_names=names,
            action_token_ids=token_ids,
            prior_token_id=prior_token_id,
            prior_action_id=prior_action_id,
            prior_response_idx=prior_response_idx,
            behavior_llm_prior_logprob=llm_logprob,
            prior_logits=logits,
            prior_log_probs=tuple(prior_log_probs),
            direct_all_action_q=direct_q,
            planner_root_mean_values=planner,
            planner_root_visit_counts=visits,
            guided_action_id=guided_action_id,
            behavior_guided_logprob=guided_logprob,
        )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "K4MCTSGuidedBehaviorRecord":
        values = _exact_mapping(
            raw,
            frozenset(cls.__dataclass_fields__),
            "K4 guided behavior",
        )
        if values["schema"] != K4_MCTS_GUIDED_BEHAVIOR_SCHEMA:
            raise ValueError(
                f"unsupported K4 guided behavior schema: {values['schema']!r}"
            )
        config = K4MCTSGuidedPolicyConfig.from_mapping(values["policy_config"])
        record = cls.build(
            action_space=values["action_space"],
            action_space_names=values["action_space_names"],
            action_token_ids=values["action_token_ids"],
            snapshot_id=values["snapshot_id"],
            prior_token_id=values["prior_token_id"],
            prior_action_id=values["prior_action_id"],
            prior_response_idx=values["prior_response_idx"],
            behavior_llm_prior_logprob=values["behavior_llm_prior_logprob"],
            prior_logits=values["prior_logits"],
            direct_all_action_q=values["direct_all_action_q"],
            planner_root_mean_values=values["planner_root_mean_values"],
            planner_root_visit_counts=values["planner_root_visit_counts"],
            guided_action_id=values["guided_action_id"],
            behavior_guided_logprob=values["behavior_guided_logprob"],
            config=config,
        )
        if values["contract_id"] != record.contract_id:
            raise ValueError("K4 guided behavior contract_id mismatch")
        recorded_prior = _finite_vector(values["prior_log_probs"], "prior_log_probs")
        if len(recorded_prior) != len(record.prior_log_probs) or any(
            not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=_logprob_tolerance(config.score_dtype, expected),
            )
            for actual, expected in zip(
                recorded_prior,
                record.prior_log_probs,
                strict=True,
            )
        ):
            raise ValueError("K4 guided behavior prior_log_probs mismatch")
        return record

    def to_mapping(self) -> dict[str, Any]:
        raw = asdict(self)
        for field in (
            "action_space_names",
            "action_token_ids",
            "prior_logits",
            "prior_log_probs",
            "direct_all_action_q",
            "planner_root_mean_values",
            "planner_root_visit_counts",
        ):
            raw[field] = list(raw[field])
        return raw

    def record_id(self) -> str:
        return _sha256_id(self.to_mapping())


def _root_visits(
    values: Sequence[int],
    *,
    action_count: int,
    expected_total: int,
) -> list[int]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("K4 planner_root_visit_counts must be a sequence")
    visits = list(values)
    if len(visits) != action_count or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in visits
    ):
        raise ValueError(
            "K4 planner_root_visit_counts must contain one positive int per action"
        )
    if sum(visits) != expected_total:
        raise ValueError(
            "K4 planner_root_visit_counts must sum to mcts_num_simulations"
        )
    return visits


def _action_table(
    action_space: str,
    action_space_names: Sequence[str],
    action_token_ids: Sequence[int],
) -> dict[str, Any]:
    if not isinstance(action_space, str) or not action_space:
        raise ValueError("K4 action_space must be non-empty")
    if isinstance(action_space_names, (str, bytes)) or not isinstance(
        action_space_names, Sequence
    ):
        raise ValueError("K4 action_space_names must be a sequence")
    names = list(action_space_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("K4 action_space_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("K4 action_space_names must be unique")
    if isinstance(action_token_ids, (str, bytes)) or not isinstance(
        action_token_ids, Sequence
    ):
        raise ValueError("K4 action_token_ids must be a sequence")
    token_ids = list(action_token_ids)
    if len(token_ids) != len(names) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in token_ids
    ):
        raise ValueError("K4 action_token_ids must align as non-negative ints")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("K4 action_token_ids must be unique")
    return {
        "action_space": action_space,
        "action_space_names": names,
        "action_token_ids": token_ids,
    }


def _log_softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    normalizer = maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    )
    return [value - normalizer for value in values]


def _finite_vector(values: Sequence[Real], field: str) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"K4 guided {field} must be a sequence")
    result = [_finite_float(value, field) for value in values]
    if not result:
        raise ValueError(f"K4 guided {field} must be non-empty")
    return result


def _finite_float(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"K4 guided {field} must be finite")
    normalized = float(value)
    return 0.0 if normalized == 0.0 else normalized


def _positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"K4 guided {field} must be a positive int")
    return value


def _logprob_tolerance(score_dtype: str, expected: float) -> float:
    epsilon = {
        "float64": 2.0**-52,
        "float32": 2.0**-23,
        "bfloat16": 2.0**-7,
    }[score_dtype]
    return 8.0 * epsilon * max(1.0, abs(expected))


def _sha256_id(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(
            dict(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


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
    "K4_MCTS_GUIDED_BEHAVIOR_SCHEMA",
    "K4MCTSGuidedBehaviorRecord",
    "K4MCTSGuidedPolicyConfig",
    "k4_guided_log_probs_reference",
    "parse_k4_mcts_joint_policy_section",
]
