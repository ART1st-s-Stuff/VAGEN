"""Standalone vLLM replica with Nimloth's per-request turn logits processor."""

from __future__ import annotations

from typing import Any

import ray

from verl.workers.rollout.replica import RolloutReplicaRegistry
from verl.workers.rollout.vllm_rollout.vllm_async_server import (
    vLLMHttpServerBase,
    vLLMReplica,
)


@ray.remote(num_cpus=1)
class NimlothVLLMHttpServer(vLLMHttpServerBase):
    """Install the shared K-slot turn processor without changing generic VERL."""

    async def launch_server(
        self,
        master_address: str | None = None,
        master_port: int | None = None,
    ) -> None:
        engine_kwargs: dict[str, Any] = self.config.engine_kwargs.setdefault(
            "vllm", {}
        )
        existing = engine_kwargs.get("logits_processors")
        expected = (
            "nimloth.backbone.qwen25vl.vllm_logits:"
            "TurnResponseLogitsProcessor"
        )
        if existing not in (None, expected):
            raise ValueError("Nimloth rollout requires its exact logits processor")
        # VERL's server CLI serializer emits one value per config key; vLLM's
        # argparse layer then normalizes this single class path into a list.
        engine_kwargs["logits_processors"] = expected
        engine_kwargs.setdefault("logprobs_mode", "processed_logprobs")
        await super().launch_server(
            master_address=master_address,
            master_port=master_port,
        )


class NimlothVLLMReplica(vLLMReplica):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.server_class = NimlothVLLMHttpServer


RolloutReplicaRegistry.register("nimloth_vllm", lambda: NimlothVLLMReplica)


__all__ = ["NimlothVLLMHttpServer", "NimlothVLLMReplica"]
