"""Coordinator-keyed action sampling from K4 MCTS Scheme-B scores."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

from .planning_contract import (
    K4MCTSGuidedPolicyConfig,
    k4_guided_log_probs_reference,
)
from .sampling import GuidedActionDrawKey

K4_MCTS_GUIDED_ACTION_DRAW_SCHEMA = "vagen_k4_mcts_guided_action_draw_v1"


@dataclass(frozen=True)
class K4MCTSGuidedActionDrawRecord:
    """One auditable root-action draw with direct-Q and planner separation."""

    schema: str
    contract_id: str
    draw_key: GuidedActionDrawKey
    policy_config: K4MCTSGuidedPolicyConfig
    action_space: str
    action_space_names: tuple[str, ...]
    action_token_ids: tuple[int, ...]
    prior_logits: tuple[float, ...]
    prior_log_probs: tuple[float, ...]
    direct_all_action_q: tuple[float, ...]
    planner_root_mean_values: tuple[float, ...]
    planner_root_visit_counts: tuple[int, ...]
    guided_log_probs: tuple[float, ...]
    uniform_draw: float
    guided_action_id: int
    behavior_guided_logprob: float

    def __post_init__(self) -> None:
        if self.schema != K4_MCTS_GUIDED_ACTION_DRAW_SCHEMA:
            raise ValueError(
                f"unsupported K4 guided action draw schema: {self.schema!r}"
            )
        key = _canonical_key(self.draw_key)
        config = _canonical_config(self.policy_config)
        action_space, names, token_ids = _action_table(
            self.action_space,
            self.action_space_names,
            self.action_token_ids,
        )
        contract_id = config.contract_id(action_space, names, token_ids)
        if self.contract_id != contract_id or key.contract_id != contract_id:
            raise ValueError("K4 action draw contract identity mismatch")
        logits = tuple(_finite_vector(self.prior_logits, "prior_logits"))
        direct_q = tuple(
            _finite_vector(self.direct_all_action_q, "direct_all_action_q")
        )
        planner = tuple(
            _finite_vector(
                self.planner_root_mean_values,
                "planner_root_mean_values",
            )
        )
        visits = tuple(
            _root_visits(
                self.planner_root_visit_counts,
                action_count=len(names),
                expected_total=config.mcts_num_simulations,
            )
        )
        if len(logits) != len(names) or len(direct_q) != len(names) or len(planner) != len(names):
            raise ValueError("K4 action draw score vectors must align with actions")
        prior, guided = k4_guided_log_probs_reference(logits, planner, config)
        recorded_prior = tuple(
            _finite_vector(self.prior_log_probs, "prior_log_probs")
        )
        recorded_guided = tuple(
            _finite_vector(self.guided_log_probs, "guided_log_probs")
        )
        if recorded_prior != tuple(prior) or recorded_guided != tuple(guided):
            raise ValueError("K4 action draw log-probs do not match scores")
        draw = _uniform_draw(self.uniform_draw)
        if draw != key.uniform_draw():
            raise ValueError("K4 action draw uniform does not match draw key")
        selected = _inverse_cdf_action(guided, draw)
        if self.guided_action_id != selected:
            raise ValueError("K4 guided_action_id does not match inverse CDF")
        behavior_logprob = _finite_float(
            self.behavior_guided_logprob,
            "behavior_guided_logprob",
        )
        if behavior_logprob != guided[selected]:
            raise ValueError("K4 behavior log-prob does not match selected action")
        object.__setattr__(self, "draw_key", key)
        object.__setattr__(self, "policy_config", config)
        object.__setattr__(self, "action_space", action_space)
        object.__setattr__(self, "action_space_names", names)
        object.__setattr__(self, "action_token_ids", token_ids)
        object.__setattr__(self, "prior_logits", logits)
        object.__setattr__(self, "prior_log_probs", recorded_prior)
        object.__setattr__(self, "direct_all_action_q", direct_q)
        object.__setattr__(self, "planner_root_mean_values", planner)
        object.__setattr__(self, "planner_root_visit_counts", visits)
        object.__setattr__(self, "guided_log_probs", recorded_guided)
        object.__setattr__(self, "uniform_draw", draw)
        object.__setattr__(self, "behavior_guided_logprob", behavior_logprob)

    @classmethod
    def build(
        cls,
        *,
        action_space: str,
        action_space_names: Sequence[str],
        action_token_ids: Sequence[int],
        prior_logits: Sequence[Real],
        direct_all_action_q: Sequence[Real],
        planner_root_mean_values: Sequence[Real],
        planner_root_visit_counts: Sequence[int],
        draw_key: GuidedActionDrawKey,
        config: K4MCTSGuidedPolicyConfig,
    ) -> "K4MCTSGuidedActionDrawRecord":
        key = _canonical_key(draw_key)
        canonical = _canonical_config(config)
        action_space, names, token_ids = _action_table(
            action_space,
            action_space_names,
            action_token_ids,
        )
        contract_id = canonical.contract_id(action_space, names, token_ids)
        if key.contract_id != contract_id:
            raise ValueError("K4 draw key contract does not match policy")
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
                expected_total=canonical.mcts_num_simulations,
            )
        )
        if len(logits) != len(names) or len(direct_q) != len(names) or len(planner) != len(names):
            raise ValueError("K4 action draw score vectors must align with actions")
        prior, guided = k4_guided_log_probs_reference(logits, planner, canonical)
        draw = key.uniform_draw()
        selected = _inverse_cdf_action(guided, draw)
        return cls(
            schema=K4_MCTS_GUIDED_ACTION_DRAW_SCHEMA,
            contract_id=contract_id,
            draw_key=key,
            policy_config=canonical,
            action_space=action_space,
            action_space_names=names,
            action_token_ids=token_ids,
            prior_logits=logits,
            prior_log_probs=tuple(prior),
            direct_all_action_q=direct_q,
            planner_root_mean_values=planner,
            planner_root_visit_counts=visits,
            guided_log_probs=tuple(guided),
            uniform_draw=draw,
            guided_action_id=selected,
            behavior_guided_logprob=guided[selected],
        )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "K4MCTSGuidedActionDrawRecord":
        if not isinstance(raw, Mapping):
            raise ValueError("K4 action draw must be a mapping")
        fields = frozenset(cls.__dataclass_fields__)
        missing = fields - set(raw)
        if missing:
            raise ValueError(f"K4 action draw is missing fields: {sorted(missing)}")
        unexpected = set(raw) - fields
        if unexpected:
            raise ValueError(
                f"K4 action draw has unexpected fields: {sorted(unexpected)}"
            )
        return cls(
            schema=raw["schema"],
            contract_id=raw["contract_id"],
            draw_key=GuidedActionDrawKey.from_mapping(raw["draw_key"]),
            policy_config=K4MCTSGuidedPolicyConfig.from_mapping(
                raw["policy_config"]
            ),
            action_space=raw["action_space"],
            action_space_names=raw["action_space_names"],
            action_token_ids=raw["action_token_ids"],
            prior_logits=raw["prior_logits"],
            prior_log_probs=raw["prior_log_probs"],
            direct_all_action_q=raw["direct_all_action_q"],
            planner_root_mean_values=raw["planner_root_mean_values"],
            planner_root_visit_counts=raw["planner_root_visit_counts"],
            guided_log_probs=raw["guided_log_probs"],
            uniform_draw=raw["uniform_draw"],
            guided_action_id=raw["guided_action_id"],
            behavior_guided_logprob=raw["behavior_guided_logprob"],
        )

    def to_mapping(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["draw_key"] = self.draw_key.to_mapping()
        for field in (
            "action_space_names",
            "action_token_ids",
            "prior_logits",
            "prior_log_probs",
            "direct_all_action_q",
            "planner_root_mean_values",
            "planner_root_visit_counts",
            "guided_log_probs",
        ):
            raw[field] = list(raw[field])
        return raw

    def record_id(self) -> str:
        payload = json.dumps(
            self.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def sample_k4_mcts_guided_action(
    *,
    action_space: str,
    action_space_names: Sequence[str],
    action_token_ids: Sequence[int],
    prior_logits: Sequence[Real],
    direct_all_action_q: Sequence[Real],
    planner_root_mean_values: Sequence[Real],
    planner_root_visit_counts: Sequence[int],
    draw_key: GuidedActionDrawKey,
    config: K4MCTSGuidedPolicyConfig,
) -> K4MCTSGuidedActionDrawRecord:
    return K4MCTSGuidedActionDrawRecord.build(
        action_space=action_space,
        action_space_names=action_space_names,
        action_token_ids=action_token_ids,
        prior_logits=prior_logits,
        direct_all_action_q=direct_all_action_q,
        planner_root_mean_values=planner_root_mean_values,
        planner_root_visit_counts=planner_root_visit_counts,
        draw_key=draw_key,
        config=config,
    )


def _inverse_cdf_action(log_probs: Sequence[float], draw: float) -> int:
    if draw == 0.0:
        return 0
    probabilities = [math.exp(value) for value in log_probs]
    for action_id in range(len(probabilities) - 1):
        if draw < math.fsum(probabilities[: action_id + 1]):
            return action_id
    return len(probabilities) - 1


def _canonical_key(value: GuidedActionDrawKey) -> GuidedActionDrawKey:
    if not isinstance(value, GuidedActionDrawKey):
        raise ValueError("K4 action draw requires GuidedActionDrawKey")
    return GuidedActionDrawKey.from_mapping(value.to_mapping())


def _canonical_config(
    value: K4MCTSGuidedPolicyConfig,
) -> K4MCTSGuidedPolicyConfig:
    if not isinstance(value, K4MCTSGuidedPolicyConfig):
        raise ValueError("K4 action draw requires K4MCTSGuidedPolicyConfig")
    return K4MCTSGuidedPolicyConfig.from_mapping(value.to_mapping())


def _action_table(
    action_space: Any,
    action_space_names: Sequence[str],
    action_token_ids: Sequence[int],
) -> tuple[str, tuple[str, ...], tuple[int, ...]]:
    if not isinstance(action_space, str) or not action_space:
        raise ValueError("K4 action draw action_space must be non-empty")
    names = tuple(action_space_names)
    token_ids = tuple(action_token_ids)
    if not names or len(token_ids) != len(names):
        raise ValueError("K4 action draw action table must be non-empty and aligned")
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("K4 action names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("K4 action names must be unique")
    if any(
        isinstance(token, bool) or not isinstance(token, int) or token < 0
        for token in token_ids
    ) or len(set(token_ids)) != len(token_ids):
        raise ValueError("K4 action token ids must be unique non-negative ints")
    return action_space, names, token_ids


def _root_visits(
    values: Sequence[int],
    *,
    action_count: int,
    expected_total: int,
) -> list[int]:
    visits = list(values)
    if len(visits) != action_count or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in visits
    ):
        raise ValueError("K4 root visits require one positive int per action")
    if sum(visits) != expected_total:
        raise ValueError(
            "K4 root visits must sum to mcts_num_simulations"
        )
    return visits


def _finite_vector(values: Sequence[Real], field: str) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"K4 action draw {field} must be a sequence")
    result = [_finite_float(value, field) for value in values]
    if not result:
        raise ValueError(f"K4 action draw {field} must be non-empty")
    return result


def _finite_float(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"K4 action draw {field} must be finite")
    normalized = float(value)
    return 0.0 if normalized == 0.0 else normalized


def _uniform_draw(value: Any) -> float:
    draw = _finite_float(value, "uniform_draw")
    if not 0.0 <= draw < 1.0:
        raise ValueError("K4 uniform_draw must satisfy 0 <= draw < 1")
    return draw


__all__ = [
    "K4_MCTS_GUIDED_ACTION_DRAW_SCHEMA",
    "K4MCTSGuidedActionDrawRecord",
    "sample_k4_mcts_guided_action",
]
