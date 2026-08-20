"""Standalone vLLM replica with Nimloth turn constraints and state capture."""

from __future__ import annotations

import asyncio
import math
from typing import Any

import ray

from verl.workers.rollout.base import register_rollout
from verl.workers.rollout.replica import RolloutReplicaRegistry, TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMHttpServerBase,
    vLLMReplica,
)

_POLICY_STATE_SCHEMA = "nimloth_policy_state_v2"
_K4_POLICY_STATE_SCHEMA = "nimloth_policy_state_k4_mcts_v1"
_K4_EXPECTED_SNAPSHOT_KEY = "nimloth_k4_expected_snapshot_id"
_K4_EXPECTED_ACTIVATION_KEY = "nimloth_k4_expected_activation_version"
_K4_CAPTURE_MCTS_TRACE_KEY = "nimloth_k4_capture_mcts_trace"
_TERMINAL_LATENT_STATE_SCHEMA = "nimloth_terminal_latent_state_v1"
_CAPTURE_MODE_KEY = "nimloth_policy_state_capture_mode"
_TERMINAL_CAPTURE_MODE = "terminal_latent_only"
_WORKER_EXTENSION = (
    "nimloth.backbone.qwen25vl.vllm_hidden."
    "PolicyStateCaptureWorkerExtension"
)


def _capture_token_ids(spec: Any) -> tuple[tuple[int, ...], int, tuple[int, ...]]:
    """Split the K-slot protocol into latent, boundary, and action identities."""

    injected = tuple(int(value) for value in spec.injected_token_ids)
    if len(injected) < 2:
        raise ValueError("Nimloth capture requires latent tokens and action_start")
    return injected[:-1], injected[-1], tuple(
        int(value) for value in spec.action_token_ids
    )


def _terminal_latent_state_payload(
    request_id: str,
    generation_id: str,
    latent_hidden: Any,
    *,
    latent_token_ids: tuple[int, ...],
) -> dict[str, Any]:
    hidden = latent_hidden.tolist()
    if (
        not isinstance(hidden, list)
        or not hidden
        or len(hidden) != len(latent_token_ids)
        or any(not isinstance(row, list) or not row for row in hidden)
        or any(not math.isfinite(float(value)) for row in hidden for value in row)
    ):
        raise RuntimeError("Nimloth vLLM returned an invalid terminal latent state")
    return {
        "schema": _TERMINAL_LATENT_STATE_SCHEMA,
        "request_id": str(request_id),
        "generation_id": str(generation_id),
        "latent_token_ids": list(latent_token_ids),
        "latent_hidden": hidden,
    }


def _policy_state_payload(
    request_id: str,
    generation_id: str,
    state: Any,
    *,
    latent_token_ids: tuple[int, ...],
    action_start_token_id: int,
    action_token_ids: tuple[int, ...],
    frozen_k4_planning: dict[str, Any] | None = None,
) -> dict[str, Any]:
    latent_hidden = state.latent_hidden.tolist()
    action_logits = state.action_logits.tolist()
    if (
        not isinstance(latent_hidden, list)
        or not latent_hidden
        or any(not isinstance(row, list) or not row for row in latent_hidden)
        or not isinstance(action_logits, list)
        or not action_logits
        or any(
            not math.isfinite(float(value))
            for row in latent_hidden
            for value in row
        )
        or any(not math.isfinite(float(value)) for value in action_logits)
    ):
        raise RuntimeError("Nimloth vLLM returned an invalid policy state")
    payload = {
        "schema": (
            _POLICY_STATE_SCHEMA
            if frozen_k4_planning is None
            else _K4_POLICY_STATE_SCHEMA
        ),
        "request_id": str(request_id),
        "generation_id": str(generation_id),
        "latent_token_ids": list(latent_token_ids),
        "action_start_token_id": int(action_start_token_id),
        "action_token_ids": list(action_token_ids),
        "latent_hidden": latent_hidden,
        "action_logits": action_logits,
    }
    if frozen_k4_planning is not None:
        payload["frozen_k4_planning"] = dict(frozen_k4_planning)
    return payload


@ray.remote(num_cpus=1)
class NimlothVLLMHttpServer(vLLMHttpServerBase):
    """Install the K-slot processor and capture state from the same generation."""

    async def install_frozen_k4_planner(
        self,
        request: dict[str, Any],
    ) -> dict[str, Any]:
        """Install a shared immutable transport before a pinned rollout batch."""

        from nimloth.backbone.qwen25vl.vllm_hidden import (
            async_install_frozen_k4_planner,
        )

        fields = {
            "transport_path",
            "expected_snapshot_id",
            "expected_source_step",
            "expected_contract_id",
            "expected_activation_version",
        }
        if not isinstance(request, dict) or set(request) != fields:
            raise ValueError("K4 planner install request fields are invalid")
        lock = getattr(self, "_nimloth_planner_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            self._nimloth_planner_lock = lock
        async with lock:
            installed = await async_install_frozen_k4_planner(
                self.engine,
                transport_path=request["transport_path"],
                expected_snapshot_id=request["expected_snapshot_id"],
                expected_source_step=request["expected_source_step"],
                expected_contract_id=request["expected_contract_id"],
                expected_activation_version=request["expected_activation_version"],
            )
        self._nimloth_planner_identity = {
            "snapshot_id": installed["snapshot_id"],
            "source_step": installed["source_step"],
            "contract_id": installed["contract_id"],
            "activation_version": installed["activation_version"],
            "transport_path": installed["transport_path"],
        }
        return dict(self._nimloth_planner_identity)

    async def launch_server(
        self,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        engine_kwargs: dict[str, Any] = self.config.engine_kwargs.setdefault(
            "vllm", {}
        )
        processor = (
            "nimloth.backbone.qwen25vl.vllm_logits:"
            "TurnResponseLogitsProcessor"
        )
        if engine_kwargs.get("logits_processors") not in (None, processor):
            raise ValueError("Nimloth rollout requires its exact logits processor")
        if engine_kwargs.get("worker_extension_cls") not in (
            None,
            _WORKER_EXTENSION,
        ):
            raise ValueError("Nimloth rollout requires its exact worker extension")
        for reserved in ("data_parallel_size", "enforce_eager"):
            if reserved in engine_kwargs:
                raise ValueError(
                    f"Nimloth rollout reserves engine_kwargs.vllm.{reserved}"
                )
        if self.config.data_parallel_size != 1:
            raise ValueError(
                "Nimloth policy-state capture currently requires data_parallel_size=1"
            )
        if not self.config.enforce_eager:
            raise ValueError("Nimloth policy-state capture requires enforce_eager=True")
        engine_kwargs["logits_processors"] = processor
        engine_kwargs["worker_extension_cls"] = _WORKER_EXTENSION
        engine_kwargs.setdefault("logprobs_mode", "processed_logprobs")
        self._nimloth_planner_lock = asyncio.Lock()
        self._nimloth_planner_identity = None
        await super().launch_server(
            master_address=master_address,
            master_port=master_port,
        )

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: list[Any] | None = None,
        session_request_id: str | None = None,
    ) -> TokenOutput:
        """Bracket one Nimloth request with request-scoped worker capture."""

        from nimloth.backbone.qwen25vl.turn_generation import TurnGenerationSpec
        from nimloth.backbone.qwen25vl.vllm_hidden import (
            async_abort_policy_state_capture_for_request,
            async_pop_latent_state_capture_for_request,
            async_pop_policy_state_capture_for_request,
            async_score_frozen_k4_planner,
            async_start_policy_state_capture_for_request,
        )

        extra_args = sampling_params.get("extra_args", {})
        spec = TurnGenerationSpec.from_extra_args(extra_args)
        if spec is None:
            return await vLLMHttpServerBase.generate(
                self,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                image_data=image_data,
            )
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Nimloth capture generation_id must be a non-empty string")
        if not isinstance(session_request_id, str) or not session_request_id:
            raise ValueError("Nimloth capture requires sticky session_request_id")
        if session_request_id == request_id:
            raise ValueError(
                "Nimloth capture generation_id must differ from session_request_id"
            )
        capture_mode = extra_args.get(_CAPTURE_MODE_KEY, "policy_and_action_logits")
        if capture_mode not in {"policy_and_action_logits", _TERMINAL_CAPTURE_MODE}:
            raise ValueError(f"unsupported Nimloth policy-state capture mode: {capture_mode!r}")
        generation_id = request_id
        session_id = session_request_id
        latent_ids, action_start_id, action_ids = _capture_token_ids(spec)
        await async_start_policy_state_capture_for_request(
            self.engine,
            request_id=request_id,
            latent_token_ids=latent_ids,
            action_start_token_id=action_start_id,
            action_token_ids=action_ids,
        )
        try:
            output = await vLLMHttpServerBase.generate(
                self,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                request_id=request_id,
                image_data=image_data,
            )
            if capture_mode == _TERMINAL_CAPTURE_MODE:
                latent_hidden = await async_pop_latent_state_capture_for_request(
                    self.engine,
                    request_id=request_id,
                )
                policy_state = _terminal_latent_state_payload(
                    session_id,
                    generation_id,
                    latent_hidden,
                    latent_token_ids=latent_ids,
                )
            else:
                state = await async_pop_policy_state_capture_for_request(
                    self.engine,
                    request_id=request_id,
                )
                if len(state.latent_hidden) != len(latent_ids):
                    raise RuntimeError(
                        "Nimloth vLLM returned the wrong latent row count: "
                        f"{len(state.latent_hidden)} != {len(latent_ids)}"
                    )
                if tuple(state.action_logits.shape) != (len(action_ids),):
                    raise RuntimeError(
                        "Nimloth vLLM returned the wrong action-logit shape: "
                        f"{tuple(state.action_logits.shape)}"
                    )
                expected_snapshot = extra_args.get(_K4_EXPECTED_SNAPSHOT_KEY)
                expected_activation = extra_args.get(_K4_EXPECTED_ACTIVATION_KEY)
                capture_mcts_trace = extra_args.get(_K4_CAPTURE_MCTS_TRACE_KEY, False)
                if not isinstance(capture_mcts_trace, bool):
                    raise ValueError("K4 MCTS trace capture flag must be bool")
                planning_score = None
                if expected_snapshot is not None or expected_activation is not None:
                    if not isinstance(expected_snapshot, str) or not expected_snapshot:
                        raise ValueError("K4 generation requires expected snapshot id")
                    if (
                        isinstance(expected_activation, bool)
                        or not isinstance(expected_activation, int)
                        or expected_activation < 0
                    ):
                        raise ValueError("K4 generation requires activation version")
                    identity = getattr(self, "_nimloth_planner_identity", None)
                    if not isinstance(identity, dict) or (
                        identity["snapshot_id"] != expected_snapshot
                        or identity["activation_version"] != expected_activation
                    ):
                        raise RuntimeError(
                            "K4 generation does not match installed planner identity"
                        )
                    async with self._nimloth_planner_lock:
                        planning_score = await async_score_frozen_k4_planner(
                            self.engine,
                            latent_hidden=state.latent_hidden,
                            expected_snapshot_id=expected_snapshot,
                            expected_activation_version=expected_activation,
                            capture_mcts_trace=capture_mcts_trace,
                        )
                policy_state = _policy_state_payload(
                    session_id,
                    generation_id,
                    state,
                    latent_token_ids=latent_ids,
                    action_start_token_id=action_start_id,
                    action_token_ids=action_ids,
                    frozen_k4_planning=planning_score,
                )
        except BaseException:
            try:
                await async_abort_policy_state_capture_for_request(
                    self.engine,
                    request_id=request_id,
                )
            except BaseException:
                pass
            raise
        return output.model_copy(update={"policy_state": policy_state})


class NimlothVLLMReplica(vLLMReplica):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.server_class = NimlothVLLMHttpServer


register_rollout(
    "nimloth_vllm",
    "async",
    "verl.workers.rollout.vllm_rollout.vLLMAsyncRollout",
)
RolloutReplicaRegistry.register("nimloth_vllm", lambda: NimlothVLLMReplica)


__all__ = ["NimlothVLLMHttpServer", "NimlothVLLMReplica"]
