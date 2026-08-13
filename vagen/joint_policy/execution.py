"""Audited separation between raw LLM evidence and an executed guided action."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .contract import GuidedPolicyBehaviorRecord

GUIDED_ACTION_EXECUTION_SCHEMA = "vagen_guided_action_execution_v2"
_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class GuidedActionExecutionRequest:
    """One validated behavior record authorizing a distinct environment action."""

    schema: str
    behavior_record_id: str
    raw_response_sha256: str
    response_trace_id: str
    behavior_record: GuidedPolicyBehaviorRecord

    def __post_init__(self) -> None:
        if self.schema != GUIDED_ACTION_EXECUTION_SCHEMA:
            raise ValueError(
                f"unsupported guided action execution schema: {self.schema!r}"
            )
        if (
            not isinstance(self.raw_response_sha256, str)
            or _SHA256_ID.fullmatch(self.raw_response_sha256) is None
        ):
            raise ValueError(
                "guided action execution raw_response_sha256 must be a "
                "canonical sha256 id"
            )
        if (
            not isinstance(self.response_trace_id, str)
            or _SHA256_ID.fullmatch(self.response_trace_id) is None
        ):
            raise ValueError(
                "guided action execution response_trace_id must be a "
                "canonical sha256 id"
            )
        if not isinstance(self.behavior_record, GuidedPolicyBehaviorRecord):
            raise ValueError(
                "guided action execution behavior_record must be "
                "GuidedPolicyBehaviorRecord"
            )
        behavior = GuidedPolicyBehaviorRecord.from_mapping(
            self.behavior_record.to_mapping()
        )
        expected_id = behavior.record_id()
        if self.behavior_record_id != expected_id:
            raise ValueError(
                "guided action execution behavior_record_id mismatch: "
                f"recorded={self.behavior_record_id!r}, expected={expected_id!r}"
            )
        object.__setattr__(self, "behavior_record", behavior)

    @property
    def prior_action_id(self) -> int:
        return self.behavior_record.prior_action_id

    @property
    def prior_action_name(self) -> str:
        return self.behavior_record.action_space_names[self.prior_action_id]

    @property
    def guided_action_id(self) -> int:
        return self.behavior_record.guided_action_id

    @property
    def guided_action_name(self) -> str:
        return self.behavior_record.action_space_names[self.guided_action_id]

    def validate_raw_response(self, raw_response: str) -> None:
        actual = _raw_response_sha256(raw_response)
        if actual != self.raw_response_sha256:
            raise ValueError(
                "guided action execution raw response identity mismatch: "
                f"recorded={self.raw_response_sha256}, actual={actual}"
            )

    @classmethod
    def from_behavior(
        cls,
        behavior: GuidedPolicyBehaviorRecord,
        *,
        raw_response: str,
        response_trace_id: str,
    ) -> "GuidedActionExecutionRequest":
        if not isinstance(behavior, GuidedPolicyBehaviorRecord):
            raise ValueError(
                "guided action execution behavior must be "
                "GuidedPolicyBehaviorRecord"
            )
        canonical = GuidedPolicyBehaviorRecord.from_mapping(behavior.to_mapping())
        return cls(
            schema=GUIDED_ACTION_EXECUTION_SCHEMA,
            behavior_record_id=canonical.record_id(),
            raw_response_sha256=_raw_response_sha256(raw_response),
            response_trace_id=response_trace_id,
            behavior_record=canonical,
        )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "GuidedActionExecutionRequest":
        if not isinstance(raw, Mapping):
            raise ValueError("guided action execution request must be a mapping")
        fields = frozenset(cls.__dataclass_fields__)
        missing = fields - set(raw)
        if missing:
            raise ValueError(
                "guided action execution request is missing fields: "
                f"{sorted(missing)}"
            )
        unexpected = set(raw) - fields
        if unexpected:
            raise ValueError(
                "guided action execution request has unexpected fields: "
                f"{sorted(unexpected)}"
            )
        if raw["schema"] != GUIDED_ACTION_EXECUTION_SCHEMA:
            raise ValueError(
                "unsupported guided action execution schema: "
                f"{raw['schema']!r}"
            )
        return cls(
            schema=GUIDED_ACTION_EXECUTION_SCHEMA,
            behavior_record_id=raw["behavior_record_id"],
            raw_response_sha256=raw["raw_response_sha256"],
            response_trace_id=raw["response_trace_id"],
            behavior_record=GuidedPolicyBehaviorRecord.from_mapping(
                raw["behavior_record"]
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["behavior_record"] = self.behavior_record.to_mapping()
        return raw


def validate_guided_action_execution_result(
    info: Mapping[str, Any],
    request: GuidedActionExecutionRequest,
) -> None:
    """Prove the environment executed exactly the authorized guided action."""

    if not isinstance(info, Mapping):
        raise ValueError("guided action execution result info must be a mapping")
    canonical = GuidedActionExecutionRequest.from_mapping(request.to_mapping())
    echo = GuidedActionExecutionRequest.from_mapping(
        info.get("guided_action_execution")
    )
    if echo != canonical:
        raise ValueError("guided action execution result request echo mismatch")
    behavior = canonical.behavior_record
    if info.get("action_space") != behavior.action_space:
        raise ValueError("guided action execution result action_space mismatch")
    if info.get("action_space_names") != list(behavior.action_space_names):
        raise ValueError("guided action execution result action table mismatch")
    if info.get("executed_action_ids") != [behavior.guided_action_id]:
        raise ValueError("guided action execution result action id mismatch")
    if info.get("executed_action_names") != [canonical.guided_action_name]:
        raise ValueError("guided action execution result action name mismatch")
    raw_response = info.get("llm_raw_response")
    if not isinstance(raw_response, str):
        raise ValueError("guided action execution result is missing raw response evidence")
    canonical.validate_raw_response(raw_response)


def _raw_response_sha256(raw_response: str) -> str:
    if not isinstance(raw_response, str) or not raw_response:
        raise ValueError("guided action execution raw_response must be non-empty")
    digest = hashlib.sha256(raw_response.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


__all__ = [
    "GUIDED_ACTION_EXECUTION_SCHEMA",
    "GuidedActionExecutionRequest",
    "validate_guided_action_execution_result",
]
