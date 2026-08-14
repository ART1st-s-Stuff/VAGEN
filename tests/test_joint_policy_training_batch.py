import math
import unittest


class _Batch:
    def __init__(self, tensors, non_tensors):
        self.batch = tensors
        self.non_tensor_batch = non_tensors
        self.meta_info = {}

    def __len__(self):
        return self.batch["responses"].shape[0]


class JointPolicyTrainingBatchTest(unittest.TestCase):
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
                "gamma": 0.9,
                "gae_lambda": 0.8,
                "ppo_clip_ratio": 0.2,
                "normalize_advantages": True,
                "token_kl_coefficient": 0.01,
                "token_kl_type": "low_var_kl",
                "guided_entropy_coefficient": 0.02,
                "checkpoint_frequency": 10,
                "actor_optimizer": {
                    "name": "adamw",
                    "lr": 1e-6,
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

    def _behavior(self, *, guided_action_id=0):
        from vagen.joint_policy.contract import (
            FrozenQGuidedPolicyConfig,
            GuidedPolicyBehaviorRecord,
            guided_log_probs_reference,
        )

        policy = FrozenQGuidedPolicyConfig.from_mapping(
            {
                "implementation": "frozen_q_guided_v1",
                "alpha": 1.0,
                "beta": 1.0,
                "prior_temperature": 1.0,
                "backprop_to_llm": True,
                "score_dtype": "float32",
            }
        )
        prior_logits = [0.0, 0.0]
        frozen_q = [1.0, 0.0]
        prior_log_probs, guided_log_probs = guided_log_probs_reference(
            prior_logits,
            frozen_q,
            policy,
        )
        return GuidedPolicyBehaviorRecord.build(
            action_space="navigation_v1",
            action_space_names=["left", "right"],
            action_token_ids=[100, 101],
            snapshot_id="snapshot-776",
            prior_token_id=100,
            prior_action_id=0,
            prior_response_idx=1,
            behavior_llm_prior_logprob=prior_log_probs[0],
            prior_logits=prior_logits,
            frozen_all_action_q=frozen_q,
            guided_action_id=guided_action_id,
            behavior_guided_logprob=guided_log_probs[guided_action_id],
            config=policy,
        )

    def _batch(self):
        import numpy as np
        import torch

        from nimloth.training.rl.joint_behavior import NimlothPolicyResponseTrace
        from nimloth.training.rl.joint_frozen_q_owner import FrozenQBatchPin
        from vagen.joint_policy.terminal_state import TerminalStateTrace

        behavior = self._behavior()
        ledger_base = {
            "schema": "vagen_decision_ledger_v2_frozen_q_guided",
            "action_space": "navigation_v1",
            "action_space_names": ["left", "right"],
            "executed_action_ids": [0],
            "executed_action_names": ["left"],
            "decision_sources": ["frozen_q_guided"],
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
        ledgers = [dict(ledger_base), dict(ledger_base)]
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
        states = []
        traces = []
        for index in range(2):
            states.append(
                {
                    "schema": "nimloth_policy_state_v2",
                    "request_id": "request-1",
                    "generation_id": f"generation-{index}",
                    "latent_token_ids": [90],
                    "action_start_token_id": 91,
                    "action_token_ids": [100, 101],
                    "latent_hidden": [[0.1 + index, 0.2]],
                    "action_logits": [0.0, 0.0],
                }
            )
            traces.append(
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
            )
        terminal = TerminalStateTrace.build(
            request_id="request-1",
            generation_id="terminal-generation",
            rollout_stop_reason="task_failure",
            raw_response="<think>real final thought</think><|latent_state|><|action_start|>",
            response_ids=[12, 90, 91],
            response_mask=[1, 0, 0],
            response_logprobs=[-0.2, 0.0, 0.0],
            latent_token_ids=[90],
            action_start_token_id=91,
            latent_hidden=[[0.5, 0.6]],
        ).to_mapping()
        return _Batch(
            {
                "responses": torch.tensor([[11, 100], [11, 100]]),
                "response_mask": torch.tensor([[1, 1], [1, 1]]),
            },
            {
                "group_idx": np.array(["group", "group"], dtype=object),
                "traj_idx": np.array([0, 0]),
                "guided_turn_index": np.array([0, 1]),
                "rollout_stop_reason": np.array(["continue", "task_failure"], dtype=object),
                "decision_ledger": np.array(ledgers, dtype=object),
                "policy_state": np.array(states, dtype=object),
                "policy_response_trace": np.array(traces, dtype=object),
                "joint_policy_batch_pin": np.array([pin, pin], dtype=object),
                "terminal_state_trace": np.array([None, terminal], dtype=object),
            },
        )

    def _dataproto(self):
        from verl import DataProto

        raw = self._batch()
        return DataProto.from_dict(
            tensors=raw.batch,
            non_tensors=raw.non_tensor_batch,
        )

    def test_compiles_strict_rollout_evidence_into_tensor_targets(self) -> None:
        import torch

        from vagen.joint_policy.training_batch import (
            joint_data_metrics,
            prepare_joint_training_batch,
        )

        batch = self._dataproto()
        targets = prepare_joint_training_batch(batch, config=self._config())
        self.assertEqual(targets.discounted_returns, (0.0, 0.0))
        self.assertEqual(batch.batch["joint_critic_hidden"].shape, (2, 1, 2))
        self.assertEqual(batch.batch["joint_action_token_ids"].tolist(), [[100, 101], [100, 101]])
        self.assertTrue(batch.batch["joint_valid_mask"].all())
        self.assertEqual(batch.meta_info["joint_snapshot_source_step"], 776)
        batch.batch["token_level_scores"] = torch.zeros(2, 2)
        batch.batch["token_level_rewards"] = torch.zeros(2, 2)
        metrics = joint_data_metrics(batch)
        self.assertEqual(metrics["joint/valid_turn_count"], 2.0)
        self.assertEqual(metrics["joint/critic_return/mean"], 0.0)

    def test_rejects_trace_token_or_terminal_identity_tampering(self) -> None:
        from vagen.joint_policy.training_batch import prepare_joint_training_batch

        batch = self._dataproto()
        batch.batch["responses"][0, 1] = 101
        with self.assertRaisesRegex(ValueError, "response IDs"):
            prepare_joint_training_batch(batch, config=self._config())
        batch = self._dataproto()
        batch.non_tensor_batch["terminal_state_trace"][0] = batch.non_tensor_batch[
            "terminal_state_trace"
        ][1]
        with self.assertRaisesRegex(ValueError, "non-final"):
            prepare_joint_training_batch(batch, config=self._config())


if __name__ == "__main__":
    unittest.main()
