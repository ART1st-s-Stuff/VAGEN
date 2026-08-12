"""Dependency-light contract for the provisional frozen-Q guided policy."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

GUIDED_BEHAVIOR_SCHEMA = "vagen_frozen_q_guided_behavior_v1"
_IMPLEMENTATION = "frozen_q_guided_v1"
_REQUIRED_CONFIG_FIELDS = frozenset(
    {
        "implementation",
        "alpha",
        "beta",
        "prior_temperature",
        "backprop_to_llm",
        "score_dtype",
    }
)


@dataclass(frozen=True)
class FrozenQGuidedPolicyConfig:
    """All currently confirmed probability and gradient semantics."""

    implementation: str
    alpha: float
    beta: float
    prior_temperature: float
    backprop_to_llm: bool
    score_dtype: str

    def __post_init__(self) -> None:
        validated = type(self)._validated_fields(
            {
                "implementation": self.implementation,
                "alpha": self.alpha,
                "beta": self.beta,
                "prior_temperature": self.prior_temperature,
                "backprop_to_llm": self.backprop_to_llm,
                "score_dtype": self.score_dtype,
            }
        )
        for field, value in validated.items():
            object.__setattr__(self, field, value)

    @classmethod
    def _validated_fields(cls, raw: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, Mapping):
            raise ValueError("joint policy config must be a mapping")
        missing = _REQUIRED_CONFIG_FIELDS - set(raw)
        if missing:
            raise ValueError(f"joint policy config is missing fields: {sorted(missing)}")
        unexpected = set(raw) - _REQUIRED_CONFIG_FIELDS
        if unexpected:
            raise ValueError(f"joint policy config has unexpected fields: {sorted(unexpected)}")
        if raw["implementation"] != _IMPLEMENTATION:
            raise ValueError(
                f"unsupported joint policy implementation: {raw['implementation']!r}"
            )

        alpha = _finite_float(raw["alpha"], "alpha")
        beta = _finite_float(raw["beta"], "beta")
        prior_temperature = _finite_float(
            raw["prior_temperature"], "prior_temperature"
        )
        if alpha <= 0.0:
            raise ValueError("joint policy alpha must be positive")
        if beta < 0.0:
            raise ValueError("joint policy beta must be non-negative")
        if prior_temperature <= 0.0:
            raise ValueError("joint policy prior_temperature must be positive")
        if raw["backprop_to_llm"] is not True:
            raise ValueError("joint policy backprop_to_llm must be true")
        if raw["score_dtype"] not in {"float32", "bfloat16", "float64"}:
            raise ValueError(
                "joint policy score_dtype must be float32, bfloat16, or float64"
            )
        return {
            "implementation": _IMPLEMENTATION,
            "alpha": alpha,
            "beta": beta,
            "prior_temperature": prior_temperature,
            "backprop_to_llm": True,
            "score_dtype": raw["score_dtype"],
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "FrozenQGuidedPolicyConfig":
        return cls(**cls._validated_fields(raw))

    def contract_id(
        self,
        action_space: str,
        action_space_names: Sequence[str],
        action_token_ids: Sequence[int],
    ) -> str:
        table = _action_table(action_space, action_space_names, action_token_ids)
        return _sha256_id({"config": asdict(self), **table})


def parse_joint_policy_section(
    raw: Mapping[str, Any],
) -> FrozenQGuidedPolicyConfig | None:
    """Parse the opt-in top-level config without creating disabled defaults."""

    if not isinstance(raw, Mapping):
        raise ValueError("joint_policy section must be a mapping")
    if "enabled" not in raw or not isinstance(raw["enabled"], bool):
        raise ValueError("joint_policy.enabled must be explicit bool")
    allowed = _REQUIRED_CONFIG_FIELDS | {"enabled"}
    unexpected = set(raw) - allowed
    if unexpected:
        raise ValueError(f"joint_policy section has unexpected fields: {sorted(unexpected)}")
    if not raw["enabled"]:
        return None
    return FrozenQGuidedPolicyConfig.from_mapping(
        {field: raw.get(field) for field in _REQUIRED_CONFIG_FIELDS}
    )


@dataclass(frozen=True)
class GuidedPolicyBehaviorRecord:
    """Immutable rollout-time values needed to replay one executed action."""

    schema: str
    contract_id: str
    policy_config: FrozenQGuidedPolicyConfig
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
    frozen_all_action_q: tuple[float, ...]
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
        frozen_all_action_q: Sequence[Real],
        guided_action_id: int,
        behavior_guided_logprob: Real,
        config: FrozenQGuidedPolicyConfig,
    ) -> "GuidedPolicyBehaviorRecord":
        table = _action_table(action_space, action_space_names, action_token_ids)
        names = table["action_space_names"]
        token_ids = table["action_token_ids"]
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise ValueError("guided policy snapshot_id must be non-empty")
        for field, value in (
            ("prior_token_id", prior_token_id),
            ("prior_action_id", prior_action_id),
            ("prior_response_idx", prior_response_idx),
            ("guided_action_id", guided_action_id),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"guided policy {field} must be a non-negative int")

        logits = _finite_vector(prior_logits, "prior_logits")
        q_values = _finite_vector(frozen_all_action_q, "frozen_all_action_q")
        if len(logits) != len(names) or len(q_values) != len(names):
            raise ValueError("guided policy logits, Q, and action table must align")
        prior_log_probs, guided_log_probs = guided_log_probs_reference(
            logits,
            q_values,
            config,
        )
        if prior_action_id >= len(names):
            raise ValueError("guided policy prior_action_id is outside action space")
        if guided_action_id >= len(names):
            raise ValueError("guided policy guided_action_id is outside action space")
        if prior_token_id != token_ids[prior_action_id]:
            raise ValueError(
                "guided policy prior token id does not match prior action mapping"
            )

        llm_logprob = _finite_float(
            behavior_llm_prior_logprob,
            "behavior_llm_prior_logprob",
        )
        expected_llm_logprob = prior_log_probs[prior_action_id]
        if not math.isclose(
            llm_logprob,
            expected_llm_logprob,
            rel_tol=0.0,
            abs_tol=_logprob_tolerance(config.score_dtype, expected_llm_logprob),
        ):
            raise ValueError(
                "behavior LLM prior log-prob does not match prior logits and action: "
                f"recorded={llm_logprob}, expected={expected_llm_logprob}"
            )
        guided_logprob = _finite_float(
            behavior_guided_logprob,
            "behavior_guided_logprob",
        )
        expected_guided_logprob = guided_log_probs[guided_action_id]
        if not math.isclose(
            guided_logprob,
            expected_guided_logprob,
            rel_tol=0.0,
            abs_tol=_logprob_tolerance(config.score_dtype, expected_guided_logprob),
        ):
            raise ValueError(
                "behavior guided log-prob does not match prior logits and frozen Q: "
                f"recorded={guided_logprob}, expected={expected_guided_logprob}"
            )

        return cls(
            schema=GUIDED_BEHAVIOR_SCHEMA,
            contract_id=config.contract_id(action_space, names, token_ids),
            policy_config=config,
            snapshot_id=snapshot_id,
            action_space=action_space,
            action_space_names=tuple(names),
            action_token_ids=tuple(token_ids),
            prior_token_id=prior_token_id,
            prior_action_id=prior_action_id,
            prior_response_idx=prior_response_idx,
            behavior_llm_prior_logprob=llm_logprob,
            prior_logits=tuple(logits),
            prior_log_probs=tuple(prior_log_probs),
            frozen_all_action_q=tuple(q_values),
            guided_action_id=guided_action_id,
            behavior_guided_logprob=guided_logprob,
        )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "GuidedPolicyBehaviorRecord":
        if not isinstance(raw, Mapping):
            raise ValueError("guided behavior record must be a mapping")
        fields = frozenset(cls.__dataclass_fields__)
        missing = fields - set(raw)
        if missing:
            raise ValueError(f"guided behavior record is missing fields: {sorted(missing)}")
        unexpected = set(raw) - fields
        if unexpected:
            raise ValueError(
                f"guided behavior record has unexpected fields: {sorted(unexpected)}"
            )
        if raw["schema"] != GUIDED_BEHAVIOR_SCHEMA:
            raise ValueError(
                f"unsupported guided behavior schema: {raw['schema']!r}"
            )
        config = FrozenQGuidedPolicyConfig.from_mapping(raw["policy_config"])
        record = cls.build(
            action_space=raw["action_space"],
            action_space_names=raw["action_space_names"],
            action_token_ids=raw["action_token_ids"],
            snapshot_id=raw["snapshot_id"],
            prior_token_id=raw["prior_token_id"],
            prior_action_id=raw["prior_action_id"],
            prior_response_idx=raw["prior_response_idx"],
            behavior_llm_prior_logprob=raw["behavior_llm_prior_logprob"],
            prior_logits=raw["prior_logits"],
            frozen_all_action_q=raw["frozen_all_action_q"],
            guided_action_id=raw["guided_action_id"],
            behavior_guided_logprob=raw["behavior_guided_logprob"],
            config=config,
        )
        if raw["contract_id"] != record.contract_id:
            raise ValueError("guided behavior contract_id does not match config and action table")
        recorded_prior_log_probs = _finite_vector(
            raw["prior_log_probs"],
            "prior_log_probs",
        )
        if len(recorded_prior_log_probs) != len(record.prior_log_probs) or any(
            not math.isclose(
                actual,
                expected,
                rel_tol=0.0,
                abs_tol=_logprob_tolerance(config.score_dtype, expected),
            )
            for actual, expected in zip(
                recorded_prior_log_probs,
                record.prior_log_probs,
                strict=True,
            )
        ):
            raise ValueError("guided behavior prior_log_probs do not match prior_logits")
        return record

    def to_mapping(self) -> dict[str, Any]:
        raw = asdict(self)
        for field in (
            "action_space_names",
            "action_token_ids",
            "prior_logits",
            "prior_log_probs",
            "frozen_all_action_q",
        ):
            raw[field] = list(raw[field])
        return raw

    def record_id(self) -> str:
        return _sha256_id(self.to_mapping())


def guided_log_probs_reference(
    prior_logits: Sequence[Real],
    frozen_q: Sequence[Real],
    config: FrozenQGuidedPolicyConfig,
) -> tuple[list[float], list[float]]:
    """Reference log-softmax used to audit rollout and tensor implementations."""

    logits = _finite_vector(prior_logits, "prior_logits")
    q_values = _finite_vector(frozen_q, "frozen_q")
    if not logits or len(logits) != len(q_values):
        raise ValueError("prior logits and frozen Q must have the same non-zero length")
    scaled_prior = _finite_vector(
        [value / config.prior_temperature for value in logits],
        "scaled_prior_logits",
    )
    guided_logits = _finite_vector(
        [
            config.alpha * prior + config.beta * q
            for prior, q in zip(scaled_prior, q_values, strict=True)
        ],
        "guided_logits",
    )
    return _log_softmax(scaled_prior), _log_softmax(guided_logits)


def _log_softmax(values: Sequence[float]) -> list[float]:
    maximum = max(values)
    log_normalizer = maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    )
    return _finite_vector(
        [value - log_normalizer for value in values],
        "log_probs",
    )


def _action_table(
    action_space: str,
    action_space_names: Sequence[str],
    action_token_ids: Sequence[int],
) -> dict[str, Any]:
    if not isinstance(action_space, str) or not action_space:
        raise ValueError("joint policy action_space must be non-empty")
    names = list(action_space_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("joint policy action_space_names must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("joint policy action_space_names must be unique")
    token_ids = list(action_token_ids)
    if len(token_ids) != len(names):
        raise ValueError("joint policy action token ids must align with action names")
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in token_ids
    ):
        raise ValueError("joint policy action_token_ids must be non-negative ints")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("joint policy action_token_ids must be unique")
    return {
        "action_space": action_space,
        "action_space_names": names,
        "action_token_ids": token_ids,
    }


def _sha256_id(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _logprob_tolerance(score_dtype: str, expected: float) -> float:
    epsilon = {
        "float64": 2.0**-52,
        "float32": 2.0**-23,
        "bfloat16": 2.0**-7,
    }[score_dtype]
    return 8.0 * epsilon * max(1.0, abs(expected))


def _finite_vector(values: Sequence[Real], field: str) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"guided policy {field} must be a sequence")
    result = [_finite_float(value, field) for value in values]
    if not result:
        raise ValueError(f"guided policy {field} must be non-empty")
    return result


def _finite_float(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"joint policy {field} must be finite")
    return float(value)


__all__ = [
    "GUIDED_BEHAVIOR_SCHEMA",
    "FrozenQGuidedPolicyConfig",
    "GuidedPolicyBehaviorRecord",
    "guided_log_probs_reference",
    "parse_joint_policy_section",
]
