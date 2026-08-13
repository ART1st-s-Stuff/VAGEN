"""Standalone vLLM replica with Nimloth turn constraints and state capture."""

from __future__ import annotations

import math
from typing import Any

import ray

from verl.workers.rollout.replica import RolloutReplicaRegistry, TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMHttpServerBase,
    vLLMReplica,
)

_POLICY_STATE_SCHEMA = "nimloth_policy_state_v1"
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


def _policy_state_payload(
    request_id: str,
    state: Any,
    *,
    latent_token_ids: tuple[int, ...],
    action_start_token_id: int,
    action_token_ids: tuple[int, ...],
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
    return {
        "schema": _POLICY_STATE_SCHEMA,
        "request_id": str(request_id),
        "latent_token_ids": list(latent_token_ids),
        "action_start_token_id": int(action_start_token_id),
        "action_token_ids": list(action_token_ids),
        "latent_hidden": latent_hidden,
        "action_logits": action_logits,
    }


@ray.remote(num_cpus=1)
class NimlothVLLMHttpServer(vLLMHttpServerBase):
    """Install the K-slot processor and capture state from the same generation."""

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
    ) -> TokenOutput:
        """Bracket one Nimloth request with request-scoped worker capture."""

        from nimloth.backbone.qwen25vl.turn_generation import TurnGenerationSpec
        from nimloth.backbone.qwen25vl.vllm_hidden import (
            async_abort_policy_state_capture_for_request,
            async_pop_policy_state_capture_for_request,
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
            state = await async_pop_policy_state_capture_for_request(
                self.engine,
                request_id=request_id,
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
        return output.model_copy(
            update={
                "policy_state": _policy_state_payload(
                    request_id,
                    state,
                    latent_token_ids=latent_ids,
                    action_start_token_id=action_start_id,
                    action_token_ids=action_ids,
                )
            }
        )


class NimlothVLLMReplica(vLLMReplica):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.server_class = NimlothVLLMHttpServer


RolloutReplicaRegistry.register("nimloth_vllm", lambda: NimlothVLLMReplica)


__all__ = ["NimlothVLLMHttpServer", "NimlothVLLMReplica"]
