from __future__ import annotations

import asyncio
import math
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

from vagen.envs.navigation.utils.nimloth_format import ACTION_NAMES, latent_state_tokens
from vagen.joint_policy import FrozenQGuidedPolicyConfig, GuidedPolicyBehaviorRecord


_ACTION_TOKEN_IDS = tuple(range(100, 108))
_RESPONSE_TRACE_ID = "sha256:" + "1" * 64


def _config() -> FrozenQGuidedPolicyConfig:
    return FrozenQGuidedPolicyConfig.from_mapping(
        {
            "implementation": "frozen_q_guided_v1",
            "alpha": 1.0,
            "beta": 1.0,
            "prior_temperature": 1.0,
            "backprop_to_llm": True,
            "score_dtype": "float64",
        }
    )


def _behavior(
    *,
    prior_action_id: int = 0,
    guided_action_id: int = 4,
) -> GuidedPolicyBehaviorRecord:
    uniform_logprob = -math.log(len(ACTION_NAMES))
    return GuidedPolicyBehaviorRecord.build(
        action_space="navigation_v1",
        action_space_names=ACTION_NAMES,
        action_token_ids=_ACTION_TOKEN_IDS,
        snapshot_id="sha256:frozen-critic-step-7",
        prior_token_id=_ACTION_TOKEN_IDS[prior_action_id],
        prior_action_id=prior_action_id,
        prior_response_idx=22,
        behavior_llm_prior_logprob=uniform_logprob,
        prior_logits=[0.0] * len(ACTION_NAMES),
        frozen_all_action_q=[0.0] * len(ACTION_NAMES),
        guided_action_id=guided_action_id,
        behavior_guided_logprob=uniform_logprob,
        config=_config(),
    )


def _raw_response(action_id: int = 0) -> str:
    return (
        "<think>real model thought</think>"
        + "".join(latent_state_tokens(16))
        + f"<|action_start|><|action_({action_id})|><|action_end|>"
    )


def _guided_info(request) -> dict[str, object]:
    return {
        "llm_raw_response": _raw_response(0),
        "action_space": "navigation_v1",
        "action_space_names": list(ACTION_NAMES),
        "executed_action_ids": [request.guided_action_id],
        "executed_action_names": [request.guided_action_name],
        "guided_action_execution": request.to_mapping(),
    }


class GuidedActionExecutionContractTest(unittest.TestCase):
    def test_request_round_trip_revalidates_behavior_identity(self) -> None:
        from vagen.joint_policy import GuidedActionExecutionRequest

        request = GuidedActionExecutionRequest.from_behavior(
            _behavior(), raw_response=_raw_response(0), response_trace_id=_RESPONSE_TRACE_ID
        )
        self.assertEqual(request.prior_action_id, 0)
        self.assertEqual(request.prior_action_name, "move_forward")
        self.assertEqual(request.guided_action_id, 4)
        self.assertEqual(request.guided_action_name, "turn_right")
        self.assertEqual(
            GuidedActionExecutionRequest.from_mapping(request.to_mapping()),
            request,
        )

        forged = request.to_mapping()
        forged["behavior_record_id"] = "sha256:forged"
        with self.assertRaisesRegex(ValueError, "behavior_record_id"):
            GuidedActionExecutionRequest.from_mapping(forged)

        with self.assertRaisesRegex(ValueError, "behavior"):
            replace(request, behavior_record=replace(_behavior(), schema="forged"))
        with self.assertRaisesRegex(ValueError, "raw response identity"):
            request.validate_raw_response(_raw_response(1))

    def test_navigation_resolution_preserves_prior_and_selects_guided_action(self) -> None:
        from vagen.envs.navigation.navigation_env import (
            _resolve_navigation_execution,
        )
        from vagen.envs.navigation.utils.parse import parse_response
        from vagen.joint_policy import GuidedActionExecutionRequest

        parsed = parse_response(
            _raw_response(0),
            prompt_format="nimloth",
            max_actions=1,
            latent_token_count=16,
        )
        request = GuidedActionExecutionRequest.from_behavior(
            _behavior(), raw_response=_raw_response(0), response_trace_id=_RESPONSE_TRACE_ID
        )
        actions, canonical = _resolve_navigation_execution(
            parsed,
            request.to_mapping(),
            prompt_format="nimloth",
        )
        self.assertEqual(parsed["actions"], ["move_forward"])
        self.assertEqual(actions, ["turn_right"])
        self.assertEqual(canonical, request)

    def test_navigation_resolution_rejects_prior_or_action_table_mismatch(self) -> None:
        from vagen.envs.navigation.navigation_env import (
            _resolve_navigation_execution,
        )
        from vagen.envs.navigation.utils.parse import parse_response
        from vagen.joint_policy import GuidedActionExecutionRequest

        parsed = parse_response(
            _raw_response(0),
            prompt_format="nimloth",
            max_actions=1,
            latent_token_count=16,
        )
        wrong_prior = GuidedActionExecutionRequest.from_behavior(
            _behavior(prior_action_id=1), raw_response=_raw_response(0), response_trace_id=_RESPONSE_TRACE_ID
        )
        with self.assertRaisesRegex(ValueError, "prior action"):
            _resolve_navigation_execution(
                parsed,
                wrong_prior.to_mapping(),
                prompt_format="nimloth",
            )

        forged = GuidedActionExecutionRequest.from_behavior(
            _behavior(), raw_response=_raw_response(0), response_trace_id=_RESPONSE_TRACE_ID
        ).to_mapping()
        forged["behavior_record"]["action_space_names"][0] = "forged"
        with self.assertRaisesRegex(ValueError, "action space|contract"):
            _resolve_navigation_execution(
                parsed,
                forged,
                prompt_format="nimloth",
            )

    def test_non_nimloth_guided_override_fails_before_environment_mutation(self) -> None:
        from vagen.envs.navigation.navigation_env import NavigationEnv
        from vagen.joint_policy import GuidedActionExecutionRequest

        env = object.__new__(NavigationEnv)
        env.cfg = SimpleNamespace(
            prompt_format="free_think",
            action_sep="|",
            max_actions_per_step=1,
            latent_token_count=16,
        )
        env._exec_action = Mock()
        request = GuidedActionExecutionRequest.from_behavior(
            _behavior(), raw_response="<think>x</think><action>move_forward</action>",
            response_trace_id=_RESPONSE_TRACE_ID,
        )
        with self.assertRaisesRegex(ValueError, "prompt_format=nimloth"):
            env._sync_step(
                "<think>x</think><action>move_forward</action>",
                guided_action_execution=request.to_mapping(),
            )
        env._exec_action.assert_not_called()

    def test_navigation_sync_step_executes_guided_action_not_raw_prior(self) -> None:
        from vagen.envs.navigation.navigation_env import NavigationEnv
        from vagen.joint_policy import GuidedActionExecutionRequest

        env = object.__new__(NavigationEnv)
        env.cfg = SimpleNamespace(
            prompt_format="nimloth",
            action_sep="|",
            max_actions_per_step=1,
            latent_token_count=16,
            max_steps=30,
            format_reward=0.0,
            per_turn_format_reward=0.0,
            success_reward=10.0,
        )
        env._step_count = 0
        env._is_format_correct = True
        env._instruction = "find the target"
        env._info = {}
        env._total_reward = 0.0
        env._t0 = 0.0
        env._controller = SimpleNamespace(
            last_event=SimpleNamespace(metadata={"lastActionSuccess": True})
        )
        executed = []
        env._agent_pos = lambda: {"x": 0.0, "z": 0.0}
        env._exec_action = executed.append
        env._is_success = lambda: False
        env._distance_to_target = lambda: 2.0
        env._render_obs = lambda init: {"obs_str": "next"}

        request = GuidedActionExecutionRequest.from_behavior(
            _behavior(), raw_response=_raw_response(0), response_trace_id=_RESPONSE_TRACE_ID
        )
        _obs, reward, done, info = env._sync_step(
            _raw_response(0),
            guided_action_execution=request.to_mapping(),
        )
        self.assertEqual(executed, [5])
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertEqual(info["actions"], ["move_forward"])
        self.assertEqual(info["executed_action_ids"], [4])
        self.assertEqual(info["executed_action_names"], ["turn_right"])
        self.assertEqual(
            info["guided_action_execution"]["behavior_record_id"],
            request.behavior_record_id,
        )

    def test_navigation_without_override_keeps_raw_action_execution(self) -> None:
        from vagen.envs.navigation.navigation_env import (
            _resolve_navigation_execution,
        )
        from vagen.envs.navigation.utils.parse import parse_response

        parsed = parse_response(
            _raw_response(0),
            prompt_format="nimloth",
            max_actions=1,
            latent_token_count=16,
        )
        actions, request = _resolve_navigation_execution(
            parsed,
            None,
            prompt_format="nimloth",
        )
        self.assertEqual(actions, ["move_forward"])
        self.assertIsNone(request)


class GuidedActionRemoteTransportTest(unittest.TestCase):
    def test_client_transports_canonical_request_with_unchanged_raw_text(self) -> None:
        from vagen.envs_remote.gym_image_env_client import GymImageEnvClient
        from vagen.joint_policy import GuidedActionExecutionRequest

        env = object.__new__(GymImageEnvClient)
        env._session_id = "session-a"
        env._check_connected = lambda _method: None
        request = GuidedActionExecutionRequest.from_behavior(
            _behavior(), raw_response=_raw_response(0), response_trace_id=_RESPONSE_TRACE_ID
        )
        env._call = AsyncMock(
            return_value=(
                {
                    "obs": "next",
                    "reward": 0.0,
                    "done": False,
                    "info": _guided_info(request),
                },
                [],
            )
        )
        asyncio.run(
            env.step(
                _raw_response(0),
                guided_action_execution=request.to_mapping(),
            )
        )
        env._call.assert_awaited_once_with(
            "step_guided",
            params={
                "action_str": _raw_response(0),
                "guided_action_execution": request.to_mapping(),
            },
        )

    def test_dispatch_rejects_guided_payload_on_ordinary_step_before_mutation(self) -> None:
        from vagen.envs_remote.handler import BaseGymHandler, SessionContext
        from vagen.joint_policy import GuidedActionExecutionRequest

        class _Handler(BaseGymHandler):
            async def create_env(self, env_config):
                raise AssertionError("not used")

        request = GuidedActionExecutionRequest.from_behavior(
            _behavior(), raw_response=_raw_response(0), response_trace_id=_RESPONSE_TRACE_ID
        )
        fake_env = SimpleNamespace(step=AsyncMock())
        handler = _Handler()
        handler._sessions["session-a"] = SessionContext(
            session_id="session-a",
            env=fake_env,
            created_at=0.0,
            last_access=0.0,
        )
        with self.assertRaisesRegex(ValueError, "step_guided"):
            asyncio.run(
                handler.call(
                    "session-a",
                    "step",
                    {
                        "action_str": _raw_response(0),
                        "guided_action_execution": request.to_mapping(),
                    },
                    [],
                )
            )
        fake_env.step.assert_not_awaited()

    def test_client_rejects_mismatched_result_echo(self) -> None:
        from vagen.envs_remote.gym_image_env_client import GymImageEnvClient
        from vagen.joint_policy import GuidedActionExecutionRequest

        request = GuidedActionExecutionRequest.from_behavior(
            _behavior(), raw_response=_raw_response(0), response_trace_id=_RESPONSE_TRACE_ID
        )
        bad_info = _guided_info(request)
        bad_info["executed_action_ids"] = [request.prior_action_id]
        env = object.__new__(GymImageEnvClient)
        env._session_id = "session-a"
        env._check_connected = lambda _method: None
        env._call = AsyncMock(
            return_value=(
                {
                    "obs": "next",
                    "reward": 0.0,
                    "done": False,
                    "info": bad_info,
                },
                [],
            )
        )
        with self.assertRaisesRegex(ValueError, "action id mismatch"):
            asyncio.run(
                env.step(
                    _raw_response(0),
                    guided_action_execution=request.to_mapping(),
                )
            )

    def test_server_rejects_environment_without_guided_capability(self) -> None:
        from vagen.envs_remote.handler import BaseGymHandler
        from vagen.joint_policy import GuidedActionExecutionRequest

        request = GuidedActionExecutionRequest.from_behavior(
            _behavior(), raw_response=_raw_response(0), response_trace_id=_RESPONSE_TRACE_ID
        )

        class _Handler(BaseGymHandler):
            async def create_env(self, env_config):
                raise AssertionError("not used")

        ordinary_step = AsyncMock()
        with self.assertRaisesRegex(ValueError, "does not support"):
            asyncio.run(
                BaseGymHandler._handle_guided_step(
                    _Handler(),
                    SimpleNamespace(env=SimpleNamespace(step=ordinary_step)),
                    {
                        "action_str": _raw_response(0),
                        "guided_action_execution": request.to_mapping(),
                    },
                )
            )
        ordinary_step.assert_not_awaited()

    def test_server_revalidates_request_before_environment_step(self) -> None:
        from vagen.envs_remote.handler import BaseGymHandler
        from vagen.joint_policy import GuidedActionExecutionRequest

        request = GuidedActionExecutionRequest.from_behavior(
            _behavior(), raw_response=_raw_response(0), response_trace_id=_RESPONSE_TRACE_ID
        )
        fake_env = SimpleNamespace(
            guided_step=AsyncMock(
                return_value=(
                    {"obs_str": "next"},
                    0.0,
                    False,
                    _guided_info(request),
                )
            )
        )
        class _Handler(BaseGymHandler):
            async def create_env(self, env_config):
                raise AssertionError("not used")

        handler = _Handler()
        result = asyncio.run(
            BaseGymHandler._handle_guided_step(
                handler,
                SimpleNamespace(env=fake_env),
                {
                    "action_str": _raw_response(0),
                    "guided_action_execution": request.to_mapping(),
                },
            )
        )
        fake_env.guided_step.assert_awaited_once_with(
            _raw_response(0),
            guided_action_execution=request.to_mapping(),
        )
        self.assertEqual(result.data["reward"], 0.0)

        forged = request.to_mapping()
        forged["behavior_record_id"] = "sha256:forged"
        with self.assertRaisesRegex(ValueError, "behavior_record_id"):
            asyncio.run(
                BaseGymHandler._handle_guided_step(
                    handler,
                    SimpleNamespace(env=fake_env),
                    {
                        "action_str": _raw_response(0),
                        "guided_action_execution": forged,
                    },
                )
            )


if __name__ == "__main__":
    unittest.main()
