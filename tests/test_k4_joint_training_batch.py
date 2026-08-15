import math
import unittest


class K4JointTrainingBatchTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import numpy  # noqa: F401
            import torch  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"tensor dependencies unavailable: {exc}")

    def _config(self):
        from vagen.joint_policy.training_contract import JointTrainingConfig

        return JointTrainingConfig.from_mapping(
            {
                "implementation": "replicated_joint_update_v1",
                "run_seed": 42,
                "gamma": 1.0,
                "gae_lambda": 0.95,
                "ppo_clip_ratio": 0.2,
                "normalize_advantages": True,
                "token_kl_coefficient": 0.01,
                "token_kl_type": "low_var_kl",
                "guided_entropy_coefficient": 0.01,
                "checkpoint_frequency": 1,
                "actor_optimizer": {
                    "name": "adamw",
                    "lr": 1e-7,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                    "grad_clip": 1.0,
                    "lr_scheduler_type": "constant",
                    "lr_warmup_steps": 0,
                    "lr_warmup_steps_ratio": 0.0,
                    "min_lr_ratio": None,
                    "num_cycles": 0.5,
                },
                "critic_checkpoint": "/tmp/id74",
                "initial_snapshot_source_step": 776,
                "critic_qwen_hidden_dim": 2,
                "critic_grid_tokens": 1,
                "critic_state_dim": 2,
                "critic_action_count": 2,
                "critic_huber_delta": 1.0,
                "critic_grad_clip": 1.0,
                "critic_optimizer": {
                    "name": "adamw",
                    "lr": 1e-4,
                    "betas": [0.9, 0.95],
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                },
            }
        )

    def _behavior(self):
        from vagen.joint_policy.planning_contract import (
            K4MCTSGuidedBehaviorRecord,
            K4MCTSGuidedPolicyConfig,
            k4_guided_log_probs_reference,
        )

        policy = K4MCTSGuidedPolicyConfig.from_mapping(
            {
                "implementation": "k4_mcts_guided_v1",
                "alpha": 1.0,
                "beta": 2.0,
                "prior_temperature": 1.0,
                "backprop_to_llm": True,
                "score_dtype": "float32",
                "planning_horizon": 4,
                "mcts_num_simulations": 8,
                "mcts_exploration_constant": 1.0,
            }
        )
        prior = [0.2, -0.1]
        planner = [0.25, -0.25]
        prior_lp, guided_lp = k4_guided_log_probs_reference(prior, planner, policy)
        return K4MCTSGuidedBehaviorRecord.build(
            action_space="navigation_v1",
            action_space_names=["left", "right"],
            action_token_ids=[100, 101],
            snapshot_id="snapshot-776",
            prior_token_id=100,
            prior_action_id=0,
            prior_response_idx=1,
            behavior_llm_prior_logprob=prior_lp[0],
            prior_logits=prior,
            direct_all_action_q=[1.5, -0.5],
            planner_root_mean_values=planner,
            planner_root_visit_counts=[4, 4],
            guided_action_id=0,
            behavior_guided_logprob=guided_lp[0],
            config=policy,
        )

    def _policy_state(self, index, behavior):
        return {
            "schema": "nimloth_policy_state_k4_mcts_v1",
            "request_id": "request-1",
            "generation_id": f"generation-{index}",
            "latent_token_ids": [90],
            "action_start_token_id": 91,
            "action_token_ids": [100, 101],
            "latent_hidden": [[0.1 + index, 0.2]],
            "action_logits": list(behavior.prior_logits),
            "frozen_k4_planning": {
                "snapshot_id": behavior.snapshot_id,
                "source_step": 776,
                "contract_id": behavior.contract_id,
                "activation_version": 0,
                "tensor_parallel_rank": 0,
                "scored": True,
                "score_dtype": "float32",
                "planning_config": {
                    "horizon": 4,
                    "num_simulations": 8,
                    "exploration_constant": 1.0,
                },
                "direct_all_action_q": list(behavior.direct_all_action_q),
                "planner_root_mean_values": list(
                    behavior.planner_root_mean_values
                ),
                "planner_root_visit_counts": list(
                    behavior.planner_root_visit_counts
                ),
                "candidate_sequences": [[0, 0, 0, 0], [1, 1, 1, 1]],
                "candidate_mean_values": [0.2, -0.2],
                "candidate_visit_counts": [4, 4],
                "planner_latency_seconds": 0.1,
            },
        }

    def _batch(self):
        import numpy as np
        import torch

        from nimloth.training.rl.joint_behavior import NimlothPolicyResponseTrace
        from nimloth.training.rl.joint_frozen_q_owner import FrozenQBatchPin
        from vagen.joint_policy.terminal_state import TerminalStateTrace

        behavior = self._behavior()
        ledger = {
            "schema": "vagen_decision_ledger_v3_k4_mcts_guided",
            "action_space": "navigation_v1",
            "action_space_names": ["left", "right"],
            "executed_action_ids": [0],
            "executed_action_names": ["left"],
            "decision_sources": ["k4_mcts_guided"],
            "decision_is_policy_sampled": [True],
            "env_turn_reward": 0.0,
            "env_terminated": False,
            "rollout_truncated": False,
            "format_valid": True,
            "snapshot_id": behavior.snapshot_id,
            "contract_id": behavior.contract_id,
            "behavior_record_id": behavior.record_id(),
            "behavior_record": behavior.to_mapping(),
        }
        ledgers = [dict(ledger), dict(ledger)]
        ledgers[1]["rollout_truncated"] = True
        pin = FrozenQBatchPin(
            schema="nimloth_frozen_q_batch_pin_v1",
            batch_id="batch-1",
            policy_step=1,
            snapshot_id=behavior.snapshot_id,
            snapshot_source_step=776,
            contract_id=behavior.contract_id,
            activation_version=0,
        ).to_mapping()
        states = [self._policy_state(index, behavior) for index in range(2)]
        traces = [
            NimlothPolicyResponseTrace(
                schema="nimloth_policy_response_trace_v1",
                request_id="request-1",
                generation_id=f"generation-{index}",
                generation_spec_id="sha256:" + "0" * 64,
                response_ids=(11, 100),
                response_mask=(1, 1),
                response_logprobs=(-0.1, -math.log(2.0)),
                raw_response="<think>real</think><|action_(0)|>",
            ).to_mapping()
            for index in range(2)
        ]
        terminal = TerminalStateTrace.build(
            request_id="request-1",
            generation_id="terminal-generation",
            rollout_stop_reason="task_failure",
            raw_response="<think>final</think><|latent_state|><|action_start|>",
            response_ids=[12, 90, 91],
            response_mask=[1, 0, 0],
            response_logprobs=[-0.2, 0.0, 0.0],
            latent_token_ids=[90],
            action_start_token_id=91,
            latent_hidden=[[0.5, 0.6]],
        ).to_mapping()
        from verl import DataProto

        return DataProto.from_dict(
            tensors={
                "responses": torch.tensor([[11, 100], [11, 100]]),
                "response_mask": torch.tensor([[1, 1], [1, 1]]),
            },
            non_tensors={
                "group_idx": np.array(["group", "group"], dtype=object),
                "traj_idx": np.array([0, 0]),
                "guided_turn_index": np.array([0, 1]),
                "rollout_stop_reason": np.array(
                    ["continue", "task_failure"], dtype=object
                ),
                "decision_ledger": np.array(ledgers, dtype=object),
                "policy_state": np.array(states, dtype=object),
                "policy_response_trace": np.array(traces, dtype=object),
                "joint_policy_batch_pin": np.array([pin, pin], dtype=object),
                "terminal_state_trace": np.array([None, terminal], dtype=object),
                "image_data": np.array([["state-0"], ["state-1"]], dtype=object),
                "terminal_image_data": np.array(
                    [None, ["terminal-state"]], dtype=object
                ),
            },
        )

    def test_compiles_separate_k4_guidance_and_direct_q_tensors(self) -> None:
        from vagen.joint_policy.training_batch import prepare_joint_training_batch

        batch = self._batch()
        prepare_joint_training_batch(batch, config=self._config())
        self.assertEqual(
            batch.batch["joint_frozen_planner_root_mean_values"].tolist(),
            [[0.25, -0.25], [0.25, -0.25]],
        )
        self.assertEqual(
            batch.batch["joint_frozen_direct_all_action_q"].tolist(),
            [[1.5, -0.5], [1.5, -0.5]],
        )
        self.assertNotIn("joint_frozen_all_action_q", batch.batch)
        self.assertEqual(batch.meta_info["joint_policy_implementation"], "k4_mcts_guided_v1")
        self.assertEqual(batch.meta_info["joint_wm_window_count"], 3)
        self.assertEqual(
            batch.batch["joint_wm_future_valid_mask"].tolist(),
            [[True, True, False, False], [True, False, False, False]],
        )
        self.assertEqual(
            batch.batch["joint_wm_future_action_ids"].tolist(),
            [[0, 0, 0, 0], [0, 0, 0, 0]],
        )
        import torch

        self.assertTrue(
            torch.allclose(
                batch.batch["joint_wm_future_hidden"][0, :2],
                torch.tensor([[[1.1, 0.2]], [[0.5, 0.6]]]),
            )
        )
        self.assertEqual(
            batch.non_tensor_batch["joint_wm_future_images"].tolist(),
            [
                ["state-1", "terminal-state", None, None],
                ["terminal-state", None, None, None],
            ],
        )

    def test_rejects_terminal_hidden_shape_tampering(self) -> None:
        from vagen.joint_policy.training_batch import prepare_joint_training_batch

        batch = self._batch()
        terminal = batch.non_tensor_batch["terminal_state_trace"][1]
        terminal["latent_hidden"] = [[0.5, 0.6, 0.7]]
        with self.assertRaisesRegex(ValueError, "terminal hidden shape"):
            prepare_joint_training_batch(batch, config=self._config())

    def test_rejects_embedded_planning_evidence_tampering(self) -> None:
        from vagen.joint_policy.training_batch import prepare_joint_training_batch

        batch = self._batch()
        batch.non_tensor_batch["policy_state"][0]["frozen_k4_planning"][
            "planner_root_mean_values"
        ][0] = 99.0
        with self.assertRaisesRegex(ValueError, "planner root means"):
            prepare_joint_training_batch(batch, config=self._config())


if __name__ == "__main__":
    unittest.main()
