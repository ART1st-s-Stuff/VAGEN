from __future__ import annotations

import asyncio
import math
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


class _FakeTensor:
    def __init__(self, values):
        self._values = values
        if values and isinstance(values[0], list):
            self.shape = (len(values), len(values[0]))
        else:
            self.shape = (len(values),)

    def __len__(self):
        return len(self._values)

    def tolist(self):
        return self._values


class _PolicyState:
    def __init__(self) -> None:
        self.latent_hidden = _FakeTensor([[1.0, 2.0], [3.0, 4.0]])
        self.action_logits = _FakeTensor(
            [float(index) for index in range(8)]
        )


class _Engine:
    async def generate(self, **kwargs):
        yield SimpleNamespace(
            request_id=kwargs["request_id"],
            outputs=[
                SimpleNamespace(
                    token_ids=[11, 12],
                    logprobs=[
                        {11: SimpleNamespace(logprob=-0.1)},
                        {12: SimpleNamespace(logprob=-0.2)},
                    ],
                )
            ],
        )

    async def list_loras(self):
        return []


class _Server:
    config = SimpleNamespace(
        max_model_len=128,
        response_length=32,
        data_parallel_size=1,
        enforce_eager=True,
        engine_kwargs={"vllm": {}},
        get=lambda _name, default=None: default,
    )
    model_config = SimpleNamespace(lora_rank=0, processor=None)
    engine = _Engine()


class NimlothPolicyStateCaptureWiringTest(unittest.TestCase):
    def test_token_output_accepts_identity_bound_policy_state(self) -> None:
        try:
            from verl.workers.rollout.replica import TokenOutput
        except ImportError as exc:
            self.skipTest(f"VERL dependencies unavailable: {exc}")

        output = TokenOutput(
            token_ids=[11],
            log_probs=[-0.5],
            policy_state={
                "schema": "nimloth_policy_state_v1",
                "request_id": "request-a",
                "latent_token_ids": [90],
                "action_start_token_id": 92,
                "action_token_ids": list(range(100, 108)),
                "latent_hidden": [[1.0, 2.0]],
                "action_logits": [float(index) for index in range(8)],
            },
        )
        self.assertEqual(output.policy_state["request_id"], "request-a")

    def test_nimloth_server_brackets_exact_request_capture(self) -> None:
        try:
            from vagen.rollout.nimloth_vllm import NimlothVLLMHttpServer
        except ImportError as exc:
            self.skipTest(f"Nimloth/VAGEN dependencies unavailable: {exc}")

        start = AsyncMock()
        pop = AsyncMock(return_value=_PolicyState())
        abort = AsyncMock()
        sampling = {
            "max_new_tokens": 2,
            "logprobs": 8,
            "extra_args": {
                "nimloth_turn_response": {
                    "close_text": "</think>",
                    "close_token_ids": [7],
                    "injected_token_ids": [90, 91, 92],
                    "action_token_ids": list(range(100, 108)),
                    "action_end_token_id": 93,
                    "forbidden_reasoning_token_ids": [],
                    "max_reasoning_tokens": 4,
                }
            },
        }
        with (
            patch(
                "nimloth.backbone.qwen25vl.vllm_hidden."
                "async_start_policy_state_capture_for_request",
                start,
            ),
            patch(
                "nimloth.backbone.qwen25vl.vllm_hidden."
                "async_pop_policy_state_capture_for_request",
                pop,
            ),
            patch(
                "nimloth.backbone.qwen25vl.vllm_hidden."
                "async_abort_policy_state_capture_for_request",
                abort,
            ),
            patch(
                "verl.workers.rollout.vllm_rollout.vllm_async_server."
                "_qwen2_5_vl_dedup_image_tokens",
                side_effect=lambda ids, _processor: ids,
            ),
        ):
            output = asyncio.run(
                NimlothVLLMHttpServer.generate.__wrapped__(
                    _Server(),
                    prompt_ids=[1, 2],
                    sampling_params=sampling,
                    request_id="request-a",
                )
            )

        start.assert_awaited_once_with(
            _Server.engine,
            request_id="request-a",
            latent_token_ids=(90, 91),
            action_start_token_id=92,
            action_token_ids=tuple(range(100, 108)),
        )
        pop.assert_awaited_once_with(_Server.engine, request_id="request-a")
        abort.assert_not_awaited()
        self.assertEqual(output.policy_state["schema"], "nimloth_policy_state_v1")
        self.assertEqual(output.policy_state["request_id"], "request-a")
        self.assertEqual(output.policy_state["latent_token_ids"], [90, 91])
        self.assertEqual(output.policy_state["action_start_token_id"], 92)
        self.assertEqual(
            output.policy_state["action_token_ids"],
            list(range(100, 108)),
        )
        self.assertEqual(len(output.policy_state["latent_hidden"]), 2)
        self.assertEqual(len(output.policy_state["action_logits"]), 8)
        self.assertNotIn("nimloth_policy_state", sampling["extra_args"])
        self.assertTrue(all(math.isfinite(value) for value in output.log_probs))

    def test_base_server_rejects_response_identity_mismatch(self) -> None:
        try:
            from verl.workers.rollout.vllm_rollout.vllm_async_server import (
                vLLMHttpServerBase,
            )
        except ImportError as exc:
            self.skipTest(f"VERL dependencies unavailable: {exc}")

        class _WrongEngine(_Engine):
            async def generate(self, **_kwargs):
                yield SimpleNamespace(
                    request_id="wrong-request",
                    outputs=[SimpleNamespace(token_ids=[11], logprobs=None)],
                )

        server = _Server()
        server.engine = _WrongEngine()
        with patch(
            "verl.workers.rollout.vllm_rollout.vllm_async_server."
            "_qwen2_5_vl_dedup_image_tokens",
            side_effect=lambda ids, _processor: ids,
        ):
            with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
                asyncio.run(
                    vLLMHttpServerBase.generate(
                        server,
                        prompt_ids=[1],
                        sampling_params={"max_new_tokens": 1},
                        request_id="request-a",
                    )
                )

    def test_nimloth_server_aborts_only_failed_request(self) -> None:
        try:
            from vagen.rollout.nimloth_vllm import NimlothVLLMHttpServer
        except ImportError as exc:
            self.skipTest(f"Nimloth/VAGEN dependencies unavailable: {exc}")

        class _FailingEngine(_Engine):
            async def generate(self, **_kwargs):
                raise RuntimeError("generation failed")
                yield  # pragma: no cover

        server = _Server()
        server.engine = _FailingEngine()
        start = AsyncMock()
        abort = AsyncMock()
        sampling = {
            "max_new_tokens": 2,
            "extra_args": {
                "nimloth_turn_response": {
                    "close_text": "</think>",
                    "close_token_ids": [7],
                    "injected_token_ids": [90, 91, 92],
                    "action_token_ids": list(range(100, 108)),
                    "action_end_token_id": 93,
                    "forbidden_reasoning_token_ids": [],
                    "max_reasoning_tokens": 4,
                }
            },
        }
        with (
            patch(
                "nimloth.backbone.qwen25vl.vllm_hidden."
                "async_start_policy_state_capture_for_request",
                start,
            ),
            patch(
                "nimloth.backbone.qwen25vl.vllm_hidden."
                "async_abort_policy_state_capture_for_request",
                abort,
            ),
            patch(
                "verl.workers.rollout.vllm_rollout.vllm_async_server."
                "_qwen2_5_vl_dedup_image_tokens",
                side_effect=lambda ids, _processor: ids,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                asyncio.run(
                    NimlothVLLMHttpServer.generate.__wrapped__(
                        server,
                        prompt_ids=[1, 2],
                        sampling_params=sampling,
                        request_id="request-b",
                    )
                )

        start.assert_awaited_once_with(
            server.engine,
            request_id="request-b",
            latent_token_ids=(90, 91),
            action_start_token_id=92,
            action_token_ids=tuple(range(100, 108)),
        )
        abort.assert_awaited_once_with(server.engine, request_id="request-b")

    def test_launch_rejects_reserved_engine_overrides(self) -> None:
        try:
            from vagen.rollout.nimloth_vllm import NimlothVLLMHttpServer
        except ImportError as exc:
            self.skipTest(f"Nimloth/VAGEN dependencies unavailable: {exc}")

        for name, value in (
            ("data_parallel_size", 2),
            ("enforce_eager", False),
        ):
            server = _Server()
            server.config = SimpleNamespace(
                data_parallel_size=1,
                enforce_eager=True,
                engine_kwargs={"vllm": {name: value}},
            )
            with self.subTest(name=name), self.assertRaisesRegex(
                ValueError,
                f"reserves engine_kwargs.vllm.{name}",
            ):
                asyncio.run(
                    NimlothVLLMHttpServer.launch_server.__wrapped__(server)
                )

    def test_launch_rejects_data_parallel_capture(self) -> None:
        try:
            from vagen.rollout.nimloth_vllm import NimlothVLLMHttpServer
        except ImportError as exc:
            self.skipTest(f"Nimloth/VAGEN dependencies unavailable: {exc}")

        server = _Server()
        server.config = SimpleNamespace(
            data_parallel_size=2,
            enforce_eager=True,
            engine_kwargs={"vllm": {}},
        )
        with self.assertRaisesRegex(ValueError, "data_parallel_size=1"):
            asyncio.run(
                NimlothVLLMHttpServer.launch_server.__wrapped__(server)
            )

    def test_agent_loop_wires_policy_state_to_turn_output(self) -> None:
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "vagen/agent_loop/gym_agent_loop_no_concat.py"
        ).read_text(encoding="utf-8")
        self.assertIn("agent_data.turn_policy_state = output.policy_state", source)
        self.assertIn('"policy_state": agent_data.turn_policy_state', source)
        manager_source = (
            Path(__file__).resolve().parents[1]
            / "vagen/agent_loop/agent_loop_no_concat.py"
        ).read_text(encoding="utf-8")
        self.assertIn('all_keys = {"policy_state"}', manager_source)


if __name__ == "__main__":
    unittest.main()
