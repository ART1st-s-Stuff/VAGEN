"""Stateless coordinator-keyed inverse-CDF guided action selection."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

from .contract import FrozenQGuidedPolicyConfig, guided_log_probs_reference

GUIDED_ACTION_DRAW_KEY_SCHEMA = "vagen_guided_action_draw_key_v1"
GUIDED_ACTION_DRAW_SCHEMA = "vagen_frozen_q_guided_action_draw_v2"


@dataclass(frozen=True)
class GuidedActionDrawKey:
    """Stable logical decision identity for a stateless uniform draw."""

    schema: str
    run_seed: int
    policy_step: int
    rollout_sample_id: str
    rollout_repeat_index: int
    turn_index: int
    is_validation: bool
    snapshot_id: str
    contract_id: str

    def __post_init__(self) -> None:
        if self.schema != GUIDED_ACTION_DRAW_KEY_SCHEMA:
            raise ValueError(f"unsupported guided action draw key schema: {self.schema!r}")
        object.__setattr__(self, "run_seed", _nonnegative_int(self.run_seed, "run_seed"))
        object.__setattr__(
            self,
            "policy_step",
            _nonnegative_int(self.policy_step, "policy_step"),
        )
        object.__setattr__(
            self,
            "rollout_sample_id",
            _nonempty_string(self.rollout_sample_id, "rollout_sample_id"),
        )
        object.__setattr__(
            self,
            "rollout_repeat_index",
            _nonnegative_int(self.rollout_repeat_index, "rollout_repeat_index"),
        )
        object.__setattr__(
            self,
            "turn_index",
            _nonnegative_int(self.turn_index, "turn_index"),
        )
        if not isinstance(self.is_validation, bool):
            raise ValueError("guided action draw key is_validation must be bool")
        object.__setattr__(
            self,
            "snapshot_id",
            _nonempty_string(self.snapshot_id, "snapshot_id"),
        )
        object.__setattr__(
            self,
            "contract_id",
            _nonempty_string(self.contract_id, "contract_id"),
        )

    @classmethod
    def build(
        cls,
        *,
        run_seed: int,
        policy_step: int,
        rollout_sample_id: str,
        rollout_repeat_index: int,
        turn_index: int,
        is_validation: bool,
        snapshot_id: str,
        contract_id: str,
    ) -> "GuidedActionDrawKey":
        return cls(
            schema=GUIDED_ACTION_DRAW_KEY_SCHEMA,
            run_seed=run_seed,
            policy_step=policy_step,
            rollout_sample_id=rollout_sample_id,
            rollout_repeat_index=rollout_repeat_index,
            turn_index=turn_index,
            is_validation=is_validation,
            snapshot_id=snapshot_id,
            contract_id=contract_id,
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "GuidedActionDrawKey":
        if not isinstance(raw, Mapping):
            raise ValueError("guided action draw key must be a mapping")
        fields = frozenset(cls.__dataclass_fields__)
        missing = fields - set(raw)
        if missing:
            raise ValueError(f"guided action draw key is missing fields: {sorted(missing)}")
        unexpected = set(raw) - fields
        if unexpected:
            raise ValueError(
                f"guided action draw key has unexpected fields: {sorted(unexpected)}"
            )
        return cls(**{field: raw[field] for field in fields})

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)

    def key_id(self) -> str:
        return f"sha256:{self._digest().hex()}"

    def uniform_draw(self) -> float:
        """Map the canonical key hash to one exact IEEE-754 53-bit fraction."""

        numerator = int.from_bytes(self._digest()[:8], "big") >> 11
        return numerator / float(1 << 53)

    def _digest(self) -> bytes:
        payload = json.dumps(
            self.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).digest()


@dataclass(frozen=True)
class GuidedActionDrawCoordinator:
    """Stateless coordinator-owned factory; no worker-local RNG state exists."""

    run_seed: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_seed", _nonnegative_int(self.run_seed, "run_seed"))

    def key_for(
        self,
        *,
        policy_step: int,
        rollout_sample_id: str,
        rollout_repeat_index: int,
        turn_index: int,
        is_validation: bool,
        snapshot_id: str,
        contract_id: str,
    ) -> GuidedActionDrawKey:
        return GuidedActionDrawKey.build(
            run_seed=self.run_seed,
            policy_step=policy_step,
            rollout_sample_id=rollout_sample_id,
            rollout_repeat_index=rollout_repeat_index,
            turn_index=turn_index,
            is_validation=is_validation,
            snapshot_id=snapshot_id,
            contract_id=contract_id,
        )


@dataclass(frozen=True)
class GuidedPolicyActionDrawRecord:
    """Auditable Scheme-B selection without owning an RNG or environment step."""

    schema: str
    contract_id: str
    draw_key: GuidedActionDrawKey
    policy_config: FrozenQGuidedPolicyConfig
    action_space: str
    action_space_names: tuple[str, ...]
    action_token_ids: tuple[int, ...]
    prior_logits: tuple[float, ...]
    frozen_all_action_q: tuple[float, ...]
    guided_log_probs: tuple[float, ...]
    uniform_draw: float
    guided_action_id: int
    behavior_guided_logprob: float

    def __post_init__(self) -> None:
        if self.schema != GUIDED_ACTION_DRAW_SCHEMA:
            raise ValueError(
                f"unsupported guided action draw schema: {self.schema!r}"
            )
        draw_key = _canonical_draw_key(self.draw_key)
        config = _canonical_config(self.policy_config)
        action_space, names, token_ids = _action_table(
            self.action_space,
            self.action_space_names,
            self.action_token_ids,
        )
        expected_contract = config.contract_id(action_space, names, token_ids)
        if self.contract_id != expected_contract or draw_key.contract_id != expected_contract:
            raise ValueError(
                "guided action draw contract_id does not match key, config, and action table"
            )
        prior_logits = tuple(_finite_vector(self.prior_logits, "prior_logits"))
        frozen_q = tuple(
            _finite_vector(self.frozen_all_action_q, "frozen_all_action_q")
        )
        if len(prior_logits) != len(names) or len(frozen_q) != len(names):
            raise ValueError(
                "guided action draw logits, Q, and action table must align"
            )
        _prior_log_probs, expected_guided = guided_log_probs_reference(
            prior_logits,
            frozen_q,
            config,
        )
        guided_log_probs = tuple(
            _finite_vector(self.guided_log_probs, "guided_log_probs")
        )
        if guided_log_probs != tuple(expected_guided):
            raise ValueError(
                "guided action draw guided_log_probs do not match prior logits and frozen Q"
            )
        draw = _uniform_draw(self.uniform_draw)
        expected_draw = draw_key.uniform_draw()
        if draw != expected_draw:
            raise ValueError(
                "guided action draw uniform_draw does not match deterministic draw key"
            )
        expected_action = _inverse_cdf_action(expected_guided, draw)
        if (
            isinstance(self.guided_action_id, bool)
            or not isinstance(self.guided_action_id, int)
            or self.guided_action_id != expected_action
        ):
            raise ValueError(
                "guided action draw guided_action_id does not match uniform_draw"
            )
        expected_logprob = expected_guided[expected_action]
        behavior_logprob = _finite_float(
            self.behavior_guided_logprob,
            "behavior_guided_logprob",
        )
        if behavior_logprob != expected_logprob:
            raise ValueError(
                "guided action draw behavior_guided_logprob does not match selected action"
            )
        object.__setattr__(self, "draw_key", draw_key)
        object.__setattr__(self, "policy_config", config)
        object.__setattr__(self, "action_space", action_space)
        object.__setattr__(self, "action_space_names", names)
        object.__setattr__(self, "action_token_ids", token_ids)
        object.__setattr__(self, "prior_logits", prior_logits)
        object.__setattr__(self, "frozen_all_action_q", frozen_q)
        object.__setattr__(self, "guided_log_probs", guided_log_probs)
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
        frozen_all_action_q: Sequence[Real],
        draw_key: GuidedActionDrawKey,
        config: FrozenQGuidedPolicyConfig,
    ) -> "GuidedPolicyActionDrawRecord":
        canonical_key = _canonical_draw_key(draw_key)
        canonical_config = _canonical_config(config)
        action_space, names, token_ids = _action_table(
            action_space,
            action_space_names,
            action_token_ids,
        )
        logits = tuple(_finite_vector(prior_logits, "prior_logits"))
        frozen_q = tuple(
            _finite_vector(frozen_all_action_q, "frozen_all_action_q")
        )
        if len(logits) != len(names) or len(frozen_q) != len(names):
            raise ValueError(
                "guided action draw logits, Q, and action table must align"
            )
        _prior_log_probs, guided_log_probs = guided_log_probs_reference(
            logits,
            frozen_q,
            canonical_config,
        )
        contract_id = canonical_config.contract_id(action_space, names, token_ids)
        if canonical_key.contract_id != contract_id:
            raise ValueError(
                "guided action draw key contract does not match distribution contract"
            )
        draw = canonical_key.uniform_draw()
        selected = _inverse_cdf_action(guided_log_probs, draw)
        return cls(
            schema=GUIDED_ACTION_DRAW_SCHEMA,
            contract_id=contract_id,
            draw_key=canonical_key,
            policy_config=canonical_config,
            action_space=action_space,
            action_space_names=names,
            action_token_ids=token_ids,
            prior_logits=logits,
            frozen_all_action_q=frozen_q,
            guided_log_probs=tuple(guided_log_probs),
            uniform_draw=draw,
            guided_action_id=selected,
            behavior_guided_logprob=guided_log_probs[selected],
        )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "GuidedPolicyActionDrawRecord":
        if not isinstance(raw, Mapping):
            raise ValueError("guided action draw record must be a mapping")
        fields = frozenset(cls.__dataclass_fields__)
        missing = fields - set(raw)
        if missing:
            raise ValueError(
                f"guided action draw record is missing fields: {sorted(missing)}"
            )
        unexpected = set(raw) - fields
        if unexpected:
            raise ValueError(
                "guided action draw record has unexpected fields: "
                f"{sorted(unexpected)}"
            )
        if raw["schema"] != GUIDED_ACTION_DRAW_SCHEMA:
            raise ValueError(
                f"unsupported guided action draw schema: {raw['schema']!r}"
            )
        return cls(
            schema=GUIDED_ACTION_DRAW_SCHEMA,
            contract_id=raw["contract_id"],
            draw_key=GuidedActionDrawKey.from_mapping(raw["draw_key"]),
            policy_config=FrozenQGuidedPolicyConfig.from_mapping(
                raw["policy_config"]
            ),
            action_space=raw["action_space"],
            action_space_names=raw["action_space_names"],
            action_token_ids=raw["action_token_ids"],
            prior_logits=raw["prior_logits"],
            frozen_all_action_q=raw["frozen_all_action_q"],
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
            "frozen_all_action_q",
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


def sample_frozen_q_guided_action(
    *,
    action_space: str,
    action_space_names: Sequence[str],
    action_token_ids: Sequence[int],
    prior_logits: Sequence[Real],
    frozen_all_action_q: Sequence[Real],
    draw_key: GuidedActionDrawKey,
    config: FrozenQGuidedPolicyConfig,
) -> GuidedPolicyActionDrawRecord:
    """Derive one exact draw from the coordinator key and select by inverse CDF."""

    return GuidedPolicyActionDrawRecord.build(
        action_space=action_space,
        action_space_names=action_space_names,
        action_token_ids=action_token_ids,
        prior_logits=prior_logits,
        frozen_all_action_q=frozen_all_action_q,
        draw_key=draw_key,
        config=config,
    )


def _inverse_cdf_action(
    guided_log_probs: Sequence[float],
    uniform_draw: float,
) -> int:
    """Use half-open probability intervals with an explicit zero-tail case."""

    if uniform_draw == 0.0:
        # Every finite log-softmax entry is mathematically positive. This keeps
        # the first interval reachable even when exp(log_probability) underflows.
        return 0
    probabilities = [math.exp(value) for value in guided_log_probs]
    for action_id in range(len(probabilities) - 1):
        cumulative = math.fsum(probabilities[: action_id + 1])
        if uniform_draw < cumulative:
            return action_id
    return len(probabilities) - 1


def _canonical_draw_key(value: GuidedActionDrawKey) -> GuidedActionDrawKey:
    if not isinstance(value, GuidedActionDrawKey):
        raise ValueError("guided action draw draw_key must be GuidedActionDrawKey")
    return GuidedActionDrawKey.from_mapping(value.to_mapping())


def _canonical_config(
    value: FrozenQGuidedPolicyConfig,
) -> FrozenQGuidedPolicyConfig:
    if not isinstance(value, FrozenQGuidedPolicyConfig):
        raise ValueError(
            "guided action draw config must be FrozenQGuidedPolicyConfig"
        )
    return FrozenQGuidedPolicyConfig.from_mapping(asdict(value))


def _action_table(
    action_space: object,
    action_space_names: Sequence[str],
    action_token_ids: Sequence[int],
) -> tuple[str, tuple[str, ...], tuple[int, ...]]:
    if not isinstance(action_space, str) or not action_space:
        raise ValueError("guided action draw action_space must be non-empty")
    if isinstance(action_space_names, (str, bytes)) or not isinstance(
        action_space_names,
        Sequence,
    ):
        raise ValueError("guided action draw action_space_names must be a sequence")
    names = tuple(action_space_names)
    if not names or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(
            "guided action draw action_space_names must contain non-empty strings"
        )
    if len(set(names)) != len(names):
        raise ValueError("guided action draw action_space_names must be unique")
    if isinstance(action_token_ids, (str, bytes)) or not isinstance(
        action_token_ids,
        Sequence,
    ):
        raise ValueError("guided action draw action_token_ids must be a sequence")
    token_ids = tuple(action_token_ids)
    if len(token_ids) != len(names):
        raise ValueError(
            "guided action draw action token ids must align with action names"
        )
    if any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in token_ids
    ):
        raise ValueError(
            "guided action draw action_token_ids must be non-negative ints"
        )
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("guided action draw action_token_ids must be unique")
    return action_space, names, token_ids


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"guided action draw key {field} must be a non-negative int")
    return value


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"guided action draw key {field} must be non-empty")
    return value


def _uniform_draw(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError("guided action draw uniform_draw must be finite real")
    draw = float(value)
    if not 0.0 <= draw < 1.0:
        raise ValueError("guided action draw uniform_draw must satisfy 0 <= draw < 1")
    return 0.0 if draw == 0.0 else draw


def _finite_vector(values: Sequence[Real], field: str) -> list[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"guided action draw {field} must be a sequence")
    result = [_finite_float(value, field) for value in values]
    if not result:
        raise ValueError(f"guided action draw {field} must be non-empty")
    return result


def _finite_float(value: object, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"guided action draw {field} must be finite")
    normalized = float(value)
    return 0.0 if normalized == 0.0 else normalized


__all__ = [
    "GUIDED_ACTION_DRAW_KEY_SCHEMA",
    "GUIDED_ACTION_DRAW_SCHEMA",
    "GuidedActionDrawCoordinator",
    "GuidedActionDrawKey",
    "GuidedPolicyActionDrawRecord",
    "sample_frozen_q_guided_action",
]
