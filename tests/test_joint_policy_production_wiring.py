from __future__ import annotations

import ast
import asyncio
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


ROOT = Path(__file__).resolve().parents[1]
GYM_LOOP = ROOT / "vagen" / "agent_loop" / "gym_agent_loop_no_concat.py"
MANAGER = ROOT / "vagen" / "agent_loop" / "agent_loop_no_concat.py"


class _Tokenizer:
    def decode(self, token_ids, *, skip_special_tokens=False):
        assert skip_special_tokens is False
        table = {
            1: "real",
            2: " thought",
            7: "</",
            8: "think>",
            90: "<|latent_state|>",
            91: "<|latent_state_1|>",
            92: "<|action_start|>",
            93: "<|action_end|>",
            100: "<|action_(0)|>",
            101: "<|action_(1)|>",
        }
        return "".join(table[token_id] for token_id in token_ids)


class _ActorMethod:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def remote(self, request):
        self.calls.append(request)
        return self.result


class _Owner:
    def __init__(self, score_result):
        self.score = _ActorMethod(score_result)


class _SyncActorMethod:
    def __init__(self, function):
        self.function = function
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.function(*args, **kwargs)


class _PromptBatch:
    def __init__(self):
        self.meta_info = {"global_steps": 9, "validate": False}
        self.non_tensor_batch = {
            "rollout_sample_id": ["sample-a", "sample-b"],
            "rollout_repeat_index": [0, 1],
            "max_turns": [2, 1],
        }

    def __len__(self):
        return 2


class JointPolicyProductionWiringTest(unittest.TestCase):
    def test_manager_pins_once_around_all_workers_and_unpins_in_finally(self) -> None:
        source = MANAGER.read_text(encoding="utf-8")
        manager_start = source.index("class AgentLoopManager")
        method_start = source.index("def generate_sequences(", manager_start)
        method_end = source.index("def _performance_metrics", method_start)
        method = source[method_start:method_end]
        self.assertIn("self.frozen_q_owner.pin_batch.remote", method)
        self.assertIn("joint_policy_batch_pin", method)
        self.assertIn("guided_action_draw_keys", method)
        self.assertIn("try:", method)
        self.assertIn("finally:", method)
        self.assertIn("self.frozen_q_owner.unpin_batch.remote", method)
        self.assertLess(
            method.index("self._pin_frozen_q_batch(prompts)"),
            method.index("worker.generate_sequences.remote"),
        )
        self.assertGreater(
            method.index("unpin_batch.remote"),
            method.index("worker.generate_sequences.remote"),
        )

    def test_manager_allocates_deterministic_keys_from_one_pinned_snapshot(self) -> None:
        try:
            from nimloth.training.rl.joint_frozen_q_owner import FrozenQBatchPin
            from vagen.agent_loop.agent_loop_no_concat import AgentLoopManager
            from vagen.joint_policy import GuidedActionDrawKey
        except ImportError as exc:
            self.skipTest(f"manager dependencies unavailable: {exc}")

        status = {
            "active_snapshot_id": "snapshot-1",
            "active_source_step": 776,
            "contract_id": "contract-1",
            "activation_version": 4,
        }

        def pin_batch(request):
            return FrozenQBatchPin(
                schema="nimloth_frozen_q_batch_pin_v1",
                batch_id=request["batch_id"],
                policy_step=request["policy_step"],
                snapshot_id=request["expected_snapshot_id"],
                snapshot_source_step=776,
                contract_id="contract-1",
                activation_version=request["expected_activation_version"],
            ).to_mapping()

        owner = SimpleNamespace(
            status=_SyncActorMethod(lambda: status),
            pin_batch=_SyncActorMethod(pin_batch),
            unpin_batch=_SyncActorMethod(lambda request: status),
        )
        manager = object.__new__(AgentLoopManager)
        manager.frozen_q_owner = owner
        manager.guided_draw_run_seed = 17
        first = _PromptBatch()
        second = _PromptBatch()
        with patch(
            "vagen.agent_loop.agent_loop_no_concat.ray.get",
            side_effect=lambda value: value,
        ):
            first_pin = manager._pin_frozen_q_batch(first)
            second_pin = manager._pin_frozen_q_batch(second)
        self.assertEqual(first_pin, second_pin)
        self.assertEqual(
            first.non_tensor_batch["guided_action_draw_keys"].tolist(),
            second.non_tensor_batch["guided_action_draw_keys"].tolist(),
        )
        keys = first.non_tensor_batch["guided_action_draw_keys"].tolist()
        self.assertEqual([len(row) for row in keys], [2, 1])
        first_key = GuidedActionDrawKey.from_mapping(keys[0][0])
        self.assertEqual(first_key.run_seed, 17)
        self.assertEqual(first_key.policy_step, 9)
        self.assertEqual(first_key.rollout_sample_id, "sample-a")
        self.assertEqual(first_key.turn_index, 0)
        self.assertEqual(first_key.snapshot_id, "snapshot-1")

    def test_manager_unpins_when_worker_generation_fails(self) -> None:
        try:
            from vagen.agent_loop.agent_loop_no_concat import AgentLoopManager
        except ImportError as exc:
            self.skipTest(f"manager dependencies unavailable: {exc}")

        pin = {"schema": "pin"}
        unpin = _SyncActorMethod(lambda request: {"open_batch_count": 0})
        owner = SimpleNamespace(unpin_batch=unpin)

        class _FailingGenerate:
            def remote(self, _chunk):
                raise RuntimeError("worker failed")

        worker = SimpleNamespace(generate_sequences=_FailingGenerate())
        prompts = SimpleNamespace(chunk=lambda _count: [object()])
        manager = object.__new__(AgentLoopManager)
        manager.config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(free_cache_engine=False)
            ),
            reward_model=SimpleNamespace(
                rollout=SimpleNamespace(free_cache_engine=False)
            ),
        )
        manager.reward_model_manager = None
        manager.frozen_q_owner = owner
        manager.agent_loop_workers = [worker]
        manager._pin_frozen_q_batch = lambda _prompts: pin
        with patch(
            "vagen.agent_loop.agent_loop_no_concat.ray.get",
            side_effect=lambda value: value,
        ):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                manager.generate_sequences(prompts)
        self.assertEqual(unpin.calls, [((pin,), {})])

    def test_dataset_sample_identity_is_restart_stable_and_unique(self) -> None:
        try:
            from vagen.gym_agent_dataset import EnvSpec, _rollout_sample_id
        except ImportError as exc:
            self.skipTest(f"dataset dependencies unavailable: {exc}")

        spec = EnvSpec(
            name="Navigation",
            n_envs=2,
            data_source="navigation",
            config={"eval_set": "base", "prompt_format": "nimloth"},
            max_turns=3,
            response_length_per_turn=128,
        )
        kwargs = {
            "spec": spec,
            "spec_idx": 0,
            "env_seed": 42,
            "data_source": "navigation",
        }
        first = _rollout_sample_id(env_idx=0, **kwargs)
        retry = _rollout_sample_id(env_idx=0, **kwargs)
        second = _rollout_sample_id(env_idx=1, **kwargs)
        self.assertEqual(first, retry)
        self.assertNotEqual(first, second)
        self.assertRegex(first, r"^sha256:[0-9a-f]{64}$")

    def test_stable_identity_is_dataset_index_not_random_uid(self) -> None:
        dataset = (ROOT / "vagen" / "gym_agent_dataset.py").read_text(encoding="utf-8")
        trainer = (ROOT / "vagen" / "ray_trainer.py").read_text(encoding="utf-8")
        self.assertIn('"rollout_sample_id":', dataset)
        self.assertIn('gen_batch.non_tensor_batch["rollout_sample_id"]', trainer)
        assign_start = trainer.index("def _assign_group_and_traj_idx")
        assign_end = trainer.index("def _post_process_no_concat_batch", assign_start)
        assign = trainer[assign_start:assign_end]
        self.assertIn('"rollout_repeat_index"', assign)
        self.assertNotIn('rollout_sample_id"] = gen_batch.non_tensor_batch["uid"]', assign)

    def test_joint_pipeline_is_between_capture_validation_and_environment_mutation(self) -> None:
        source = GYM_LOOP.read_text(encoding="utf-8")
        generating_start = source.index("async def _handle_generating_state")
        env_start = source.index("async def _handle_env_state", generating_start)
        generating = source[generating_start:env_start]
        self.assertIn("_build_joint_guided_execution", generating)
        self.assertLess(
            generating.index('agent_data.turn_policy_state = output.policy_state'),
            generating.index("_build_joint_guided_execution"),
        )
        env_end = source.index("return AgentState.PENDING", env_start)
        env_method = source[env_start:env_end]
        self.assertIn("guided_action_execution", env_method)
        self.assertIn("build_guided_decision_ledger", env_method)
        self.assertIn("validate_guided_action_execution_result", env_method)
        self.assertIn('"frozen_q_scoring"', env_method)
        self.assertIn('"guided_action_draw"', env_method)
        self.assertIn('"guided_action_execution"', env_method)

    def test_pipeline_builds_complete_execution_without_mutating_environment(self) -> None:
        try:
            from nimloth.backbone.qwen25vl.turn_generation import TurnGenerationSpec
            from nimloth.training.rl.joint_critic import (
                JointActionValueCritic,
                create_frozen_critic_snapshot,
            )
            from nimloth.training.rl.joint_frozen_q_owner import (
                FrozenQBatchPin,
                FrozenQOwnerScoringResult,
            )
            from nimloth.training.rl.joint_scoring import score_captured_policy_state
            from nimloth.wm.grid import SharedSlotProjector
            from nimloth.wm.value_head import ValueHead
            from vagen.agent_loop.gym_agent_loop_no_concat import (
                AgentData,
                GymAgentLoop,
                _build_joint_guided_execution,
            )
            from vagen.joint_policy import (
                FrozenQGuidedPolicyConfig,
                GuidedActionDrawCoordinator,
            )
            import torch
        except ImportError as exc:
            self.skipTest(f"joint pipeline dependencies unavailable: {exc}")

        config = FrozenQGuidedPolicyConfig.from_mapping(
            {
                "implementation": "frozen_q_guided_v1",
                "alpha": 1.0,
                "beta": 1.0,
                "prior_temperature": 1.0,
                "backprop_to_llm": True,
                "score_dtype": "float64",
            }
        )
        spec = TurnGenerationSpec(
            close_text="</think>",
            close_token_ids=(7, 8),
            injected_token_ids=(90, 91, 92),
            action_token_ids=(100, 101),
            action_end_token_id=93,
            forbidden_reasoning_token_ids=(),
            max_reasoning_tokens=4,
        )
        critic = JointActionValueCritic(
            state_projector=SharedSlotProjector(
                input_dim=3,
                output_dim=2,
                hidden_dim=5,
                grid_tokens=2,
            ),
            value_head=ValueHead(emb_dim=2, num_actions=2, hidden_dim=4),
        ).double()
        snapshot = create_frozen_critic_snapshot(
            critic,
            source_step=776,
            contract_id=config.contract_id(
                "navigation_v1",
                ("move_forward", "turn_right"),
                spec.action_token_ids,
            ),
            score_dtype="float64",
        )
        policy_state = {
            "schema": "nimloth_policy_state_v2",
            "request_id": "episode-1",
            "generation_id": "generation-1",
            "latent_token_ids": [90, 91],
            "action_start_token_id": 92,
            "action_token_ids": [100, 101],
            "latent_hidden": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            "action_logits": [0.0, 0.0],
        }
        scoring = score_captured_policy_state(
            policy_state,
            snapshot=snapshot,
            expected_request_id="episode-1",
            expected_generation_id="generation-1",
            expected_latent_token_ids=(90, 91),
            expected_action_start_token_id=92,
            expected_action_token_ids=(100, 101),
            expected_contract_id=snapshot.contract_id,
        )
        pin = FrozenQBatchPin(
            schema="nimloth_frozen_q_batch_pin_v1",
            batch_id="batch-1",
            policy_step=1,
            snapshot_id=snapshot.snapshot_id,
            snapshot_source_step=776,
            contract_id=snapshot.contract_id,
            activation_version=4,
        )
        owner = _Owner(
            FrozenQOwnerScoringResult(
                schema="nimloth_frozen_q_owner_score_result_v1",
                batch_pin=pin,
                scoring_record=scoring,
            ).to_mapping()
        )
        draw_key = GuidedActionDrawCoordinator(17).key_for(
            policy_step=1,
            rollout_sample_id="sample-9",
            rollout_repeat_index=2,
            turn_index=0,
            is_validation=False,
            snapshot_id=snapshot.snapshot_id,
            contract_id=snapshot.contract_id,
        )
        result = asyncio.run(
            _build_joint_guided_execution(
                frozen_q_owner=owner,
                batch_pin=pin.to_mapping(),
                expected_draw_key=draw_key.to_mapping(),
                policy_config=config,
                policy_state=policy_state,
                response_ids=(1, 2, 7, 8, 90, 91, 92, 100, 93),
                response_mask=(1, 1, 1, 1, 0, 0, 0, 1, 0),
                response_logprobs=(-0.1, -0.2, -0.3, -0.4, 0.0, 0.0, 0.0, math.log(0.5), 0.0),
                raw_response=(
                    "<think>real thought</think><|latent_state|>"
                    "<|latent_state_1|><|action_start|>"
                    "<|action_(0)|><|action_end|>"
                ),
                generation_spec=spec,
                tokenizer=_Tokenizer(),
                action_space="navigation_v1",
                action_space_names=("move_forward", "turn_right"),
            )
        )
        self.assertEqual(len(owner.score.calls), 1)
        self.assertEqual(result.scoring_record.snapshot_id, snapshot.snapshot_id)
        self.assertEqual(result.action_draw.draw_key.policy_step, 1)
        self.assertEqual(result.action_draw.draw_key.rollout_sample_id, "sample-9")
        self.assertEqual(result.action_draw.draw_key.rollout_repeat_index, 2)
        self.assertEqual(result.action_draw.draw_key.turn_index, 0)
        self.assertEqual(
            result.execution.action_draw_record_id,
            result.action_draw.record_id(),
        )
        self.assertEqual(
            result.execution.response_trace_id,
            result.response_trace.trace_id(),
        )

        info = {
            "guided_action_execution": result.execution.to_mapping(),
            "action_space": result.execution.behavior_record.action_space,
            "action_space_names": list(
                result.execution.behavior_record.action_space_names
            ),
            "executed_action_ids": [result.execution.guided_action_id],
            "executed_action_names": [result.execution.guided_action_name],
            "llm_raw_response": result.response_trace.raw_response,
            "format_correct": True,
            "success": False,
        }
        env = SimpleNamespace(
            guided_step=AsyncMock(
                return_value=({"obs_str": "next"}, 0.25, False, info)
            )
        )
        agent_data = AgentData(
            metrics={},
            request_id="episode-1",
            env=env,
            response_limit=32,
            env_name="FakeNavigation",
            sys_images=[],
            cur_images=[],
            group_idx="group-1",
            traj_idx=2,
        )
        agent_data.turn_prompt_ids = [50, *result.response_trace.response_ids]
        agent_data.turn_response_ids = list(result.response_trace.response_ids)
        agent_data.turn_response_mask = list(result.response_trace.response_mask)
        agent_data.turn_response_logprobs = list(
            result.response_trace.response_logprobs
        )
        agent_data.turn_policy_state = policy_state
        agent_data.turn_guided_artifacts = result
        agent_data.last_assistant_text = result.response_trace.raw_response
        loop = SimpleNamespace(
            decision_ledger_enabled=True,
            env_max_turns=1,
            response_length=32,
            prompt_length=32,
        )
        state = asyncio.run(
            GymAgentLoop._handle_env_state(loop, agent_data)
        )
        env.guided_step.assert_awaited_once()
        self.assertEqual(state.value, "terminated")
        output = agent_data.outputs[0]
        self.assertEqual(
            output.extra_fields["guided_action_execution"],
            result.execution.to_mapping(),
        )
        self.assertEqual(
            output.extra_fields["decision_ledger"]["executed_action_ids"],
            [result.execution.guided_action_id],
        )
        self.assertEqual(
            output.extra_fields["frozen_q_scoring"]["snapshot_id"],
            snapshot.snapshot_id,
        )


if __name__ == "__main__":
    unittest.main()
