"""Immutable real-CoT/K-slot evidence for an outcome observation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any

TERMINAL_STATE_TRACE_SCHEMA = "vagen_terminal_state_trace_v1"
_OUTCOMES = frozenset({"success", "task_failure", "environment_failure"})


@dataclass(frozen=True)
class TerminalStateTrace:
    """Same-generation terminal observation state with no sampled action token."""

    schema: str
    request_id: str
    generation_id: str
    rollout_stop_reason: str
    raw_response: str
    response_ids: tuple[int, ...]
    response_mask: tuple[int, ...]
    response_logprobs: tuple[float, ...]
    latent_token_ids: tuple[int, ...]
    action_start_token_id: int
    latent_hidden: tuple[tuple[float, ...], ...]

    @classmethod
    def build(
        cls,
        *,
        request_id: str,
        generation_id: str,
        rollout_stop_reason: str,
        raw_response: str,
        response_ids: Sequence[int],
        response_mask: Sequence[int | bool],
        response_logprobs: Sequence[Real],
        latent_token_ids: Sequence[int],
        action_start_token_id: int,
        latent_hidden: Sequence[Sequence[Real]],
    ) -> "TerminalStateTrace":
        request = _identity(request_id, "request_id")
        generation = _identity(generation_id, "generation_id")
        if request == generation:
            raise ValueError("terminal request and generation identities must differ")
        if rollout_stop_reason not in _OUTCOMES:
            raise ValueError("terminal state trace requires a final outcome stop reason")
        if not isinstance(raw_response, str):
            raise ValueError("terminal raw_response must be str")
        thought_start = raw_response.find("<think>")
        thought_end = raw_response.find("</think>", thought_start + len("<think>"))
        if (
            thought_start != 0
            or thought_end < 0
            or not raw_response[len("<think>") : thought_end].strip()
        ):
            raise ValueError("terminal state trace requires model-generated real CoT")

        ids = _int_vector(response_ids, "response_ids")
        mask = _mask_vector(response_mask)
        logprobs = _float_vector(response_logprobs, "response_logprobs")
        latent_ids = _int_vector(latent_token_ids, "latent_token_ids")
        if not latent_ids or len(set(latent_ids)) != len(latent_ids):
            raise ValueError("terminal latent_token_ids must be non-empty and unique")
        action_start = _nonnegative_int(
            action_start_token_id,
            "action_start_token_id",
        )
        if action_start in latent_ids:
            raise ValueError("terminal action_start must differ from latent token ids")
        if len(ids) != len(mask) or len(ids) != len(logprobs):
            raise ValueError("terminal response IDs, mask, and log-probs must align")
        expected_suffix = (*latent_ids, action_start)
        if len(ids) <= len(expected_suffix) or tuple(ids[-len(expected_suffix) :]) != expected_suffix:
            raise ValueError(
                "terminal response must end at action_start after the exact latent suffix"
            )
        forced_start = len(ids) - len(expected_suffix)
        if any(mask[index] != 0 for index in range(forced_start, len(mask))):
            raise ValueError("terminal latent/action_start protocol tokens must be forced")
        if not any(mask[:forced_start]):
            raise ValueError("terminal state trace requires sampled real CoT tokens")

        hidden_rows = _matrix(latent_hidden, "latent_hidden")
        if len(hidden_rows) != len(latent_ids):
            raise ValueError("terminal latent hidden rows must align with latent token ids")
        return cls(
            schema=TERMINAL_STATE_TRACE_SCHEMA,
            request_id=request,
            generation_id=generation,
            rollout_stop_reason=rollout_stop_reason,
            raw_response=raw_response,
            response_ids=tuple(ids),
            response_mask=tuple(mask),
            response_logprobs=tuple(logprobs),
            latent_token_ids=tuple(latent_ids),
            action_start_token_id=action_start,
            latent_hidden=tuple(tuple(row) for row in hidden_rows),
        )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TerminalStateTrace":
        if not isinstance(raw, Mapping):
            raise ValueError("terminal state trace must be a mapping")
        fields = frozenset(cls.__dataclass_fields__)
        missing = fields - set(raw)
        if missing:
            raise ValueError(f"terminal state trace is missing fields: {sorted(missing)}")
        unexpected = set(raw) - fields
        if unexpected:
            raise ValueError(
                f"terminal state trace has unexpected fields: {sorted(unexpected)}"
            )
        if raw["schema"] != TERMINAL_STATE_TRACE_SCHEMA:
            raise ValueError(
                f"unsupported terminal state trace schema: {raw['schema']!r}"
            )
        return cls.build(
            request_id=raw["request_id"],
            generation_id=raw["generation_id"],
            rollout_stop_reason=raw["rollout_stop_reason"],
            raw_response=raw["raw_response"],
            response_ids=raw["response_ids"],
            response_mask=raw["response_mask"],
            response_logprobs=raw["response_logprobs"],
            latent_token_ids=raw["latent_token_ids"],
            action_start_token_id=raw["action_start_token_id"],
            latent_hidden=raw["latent_hidden"],
        )

    def to_mapping(self) -> dict[str, Any]:
        raw = asdict(self)
        for field in (
            "response_ids",
            "response_mask",
            "response_logprobs",
            "latent_token_ids",
            "latent_hidden",
        ):
            raw[field] = [list(value) if isinstance(value, tuple) else value for value in raw[field]]
        return raw

    def record_id(self) -> str:
        payload = json.dumps(
            self.to_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _identity(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"terminal state {field} must be non-empty str")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"terminal state {field} must be non-negative int")
    return value


def _plain_sequence(value: Any, field: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence):
        raise ValueError(f"terminal state {field} must be a plain sequence")
    return value


def _int_vector(value: Any, field: str) -> list[int]:
    result = [
        _nonnegative_int(item, field)
        for item in _plain_sequence(value, field)
    ]
    if not result:
        raise ValueError(f"terminal state {field} must be non-empty")
    return result


def _mask_vector(value: Any) -> list[int]:
    result = []
    for item in _plain_sequence(value, "response_mask"):
        if isinstance(item, bool):
            result.append(int(item))
        elif isinstance(item, int) and item in (0, 1):
            result.append(item)
        else:
            raise ValueError("terminal state response_mask must contain 0/1")
    return result


def _float_vector(value: Any, field: str) -> list[float]:
    result = []
    for item in _plain_sequence(value, field):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise ValueError(f"terminal state {field} must contain finite reals")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"terminal state {field} must contain finite values")
        result.append(0.0 if number == 0.0 else number)
    return result


def _matrix(value: Any, field: str) -> list[list[float]]:
    rows = [
        _float_vector(row, field)
        for row in _plain_sequence(value, field)
    ]
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError(f"terminal state {field} must be a non-empty rectangular matrix")
    return rows


__all__ = ["TERMINAL_STATE_TRACE_SCHEMA", "TerminalStateTrace"]
