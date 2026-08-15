"""Environment authorization for one K4 MCTS-guided root action."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from .planning_contract import K4MCTSGuidedBehaviorRecord

K4_GUIDED_ACTION_EXECUTION_SCHEMA = "vagen_k4_mcts_guided_action_execution_v1"
_SHA256_ID = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class K4MCTSGuidedActionExecutionRequest:
    """One validated K4 behavior record authorizing one real root action."""

    schema: str
    behavior_record_id: str
    raw_response_sha256: str
    response_trace_id: str
    action_draw_record_id: str
    behavior_record: K4MCTSGuidedBehaviorRecord

    def __post_init__(self) -> None:
        if self.schema != K4_GUIDED_ACTION_EXECUTION_SCHEMA:
            raise ValueError(
                f"unsupported K4 guided execution schema: {self.schema!r}"
            )
        for field in (
            "behavior_record_id",
            "raw_response_sha256",
            "response_trace_id",
            "action_draw_record_id",
        ):
            value = getattr(self, field)
            if not isinstance(value, str) or _SHA256_ID.fullmatch(value) is None:
                raise ValueError(f"K4 guided execution {field} must be sha256 id")
        if not isinstance(self.behavior_record, K4MCTSGuidedBehaviorRecord):
            raise ValueError("K4 guided execution requires K4 behavior record")
        behavior = K4MCTSGuidedBehaviorRecord.from_mapping(
            self.behavior_record.to_mapping()
        )
        if self.behavior_record_id != behavior.record_id():
            raise ValueError("K4 guided execution behavior record id mismatch")
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
        if _raw_response_sha256(raw_response) != self.raw_response_sha256:
            raise ValueError("K4 guided execution raw response identity mismatch")

    @classmethod
    def from_behavior(
        cls,
        behavior: K4MCTSGuidedBehaviorRecord,
        *,
        raw_response: str,
        response_trace_id: str,
        action_draw_record_id: str,
    ) -> "K4MCTSGuidedActionExecutionRequest":
        canonical = K4MCTSGuidedBehaviorRecord.from_mapping(behavior.to_mapping())
        return cls(
            schema=K4_GUIDED_ACTION_EXECUTION_SCHEMA,
            behavior_record_id=canonical.record_id(),
            raw_response_sha256=_raw_response_sha256(raw_response),
            response_trace_id=response_trace_id,
            action_draw_record_id=action_draw_record_id,
            behavior_record=canonical,
        )

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
    ) -> "K4MCTSGuidedActionExecutionRequest":
        if not isinstance(raw, Mapping):
            raise ValueError("K4 guided execution must be a mapping")
        fields = frozenset(cls.__dataclass_fields__)
        if set(raw) != fields:
            raise ValueError("K4 guided execution fields are invalid")
        return cls(
            schema=raw["schema"],
            behavior_record_id=raw["behavior_record_id"],
            raw_response_sha256=raw["raw_response_sha256"],
            response_trace_id=raw["response_trace_id"],
            action_draw_record_id=raw["action_draw_record_id"],
            behavior_record=K4MCTSGuidedBehaviorRecord.from_mapping(
                raw["behavior_record"]
            ),
        )

    def to_mapping(self) -> dict[str, Any]:
        raw = asdict(self)
        raw["behavior_record"] = self.behavior_record.to_mapping()
        return raw


def parse_guided_action_execution_request(
    raw: Mapping[str, Any],
) -> Any:
    """Parse either legacy direct-Q or K4 execution without schema ambiguity."""

    if not isinstance(raw, Mapping):
        raise ValueError("guided action execution must be a mapping")
    if raw.get("schema") == K4_GUIDED_ACTION_EXECUTION_SCHEMA:
        return K4MCTSGuidedActionExecutionRequest.from_mapping(raw)
    from .execution import GuidedActionExecutionRequest

    return GuidedActionExecutionRequest.from_mapping(raw)


def validate_k4_guided_action_execution_result(
    info: Mapping[str, Any],
    request: K4MCTSGuidedActionExecutionRequest,
) -> None:
    if not isinstance(info, Mapping):
        raise ValueError("K4 guided execution result info must be a mapping")
    canonical = K4MCTSGuidedActionExecutionRequest.from_mapping(request.to_mapping())
    echo = K4MCTSGuidedActionExecutionRequest.from_mapping(
        info.get("guided_action_execution")
    )
    if echo != canonical:
        raise ValueError("K4 guided execution result echo mismatch")
    behavior = canonical.behavior_record
    if info.get("action_space") != behavior.action_space:
        raise ValueError("K4 guided execution action_space mismatch")
    if info.get("action_space_names") != list(behavior.action_space_names):
        raise ValueError("K4 guided execution action table mismatch")
    if info.get("executed_action_ids") != [behavior.guided_action_id]:
        raise ValueError("K4 guided execution action id mismatch")
    if info.get("executed_action_names") != [canonical.guided_action_name]:
        raise ValueError("K4 guided execution action name mismatch")
    raw_response = info.get("llm_raw_response")
    if not isinstance(raw_response, str):
        raise ValueError("K4 guided execution result lacks raw response")
    canonical.validate_raw_response(raw_response)


def _raw_response_sha256(raw_response: str) -> str:
    if not isinstance(raw_response, str) or not raw_response:
        raise ValueError("K4 guided raw_response must be non-empty")
    return f"sha256:{hashlib.sha256(raw_response.encode('utf-8')).hexdigest()}"


__all__ = [
    "K4_GUIDED_ACTION_EXECUTION_SCHEMA",
    "K4MCTSGuidedActionExecutionRequest",
    "parse_guided_action_execution_request",
    "validate_k4_guided_action_execution_result",
]
