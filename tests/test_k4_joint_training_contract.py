import math
import unittest


_ACTIONS = ("a0", "a1")
_TOKENS = (101, 102)
_SNAPSHOT = "sha256:" + "1" * 64


def _training_config():
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


def _wm_config():
    from vagen.joint_policy.k4_training_contract import (
        K4WorldModelTrainingConfig,
    )

    return K4WorldModelTrainingConfig.from_mapping(
        {
            "implementation": "k4_world_model_update_v1",
            "planning_checkpoint": "/tmp/id74",
            "snapshot_transport_root": "/tmp/snapshots",
            "prediction_horizon": 4,
            "minimum_window_depth": 1,
            "maximum_window_depth": 4,
            "state_mse_weight": 1.0,
            "dino_grid_weight": 0.5,
            "sigreg_weight": 0.1,
            "sigreg_knots": 17,
            "sigreg_num_proj": 1024,
            "dino_identity": {
                "source": "facebook/dinov2-large",
                "revision": "47b73eefe95e8d44ec3623f8890bd894b6ea2d6c",
                "processor_fingerprint": "7d65a7de8788e87d",
                "hidden_size": 1024,
                "grid_size": 4,
            },
            "selected_action_huber_delta": 1.0,
            "grad_clip": 1.0,
            "optimizer": {
                "name": "adamw",
                "projector_lr": 1e-4,
                "predictor_lr": 1e-4,
                "value_head_lr": 1e-4,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        }
    )


def _behavior():
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
    prior = (0.4, -0.2)
    planner = (-0.5, 0.5)
    direct_q = (10.0, -2.0)
    prior_lp, guided_lp = k4_guided_log_probs_reference(prior, planner, policy)
    return K4MCTSGuidedBehaviorRecord.build(
        action_space="test_v1",
        action_space_names=_ACTIONS,
        action_token_ids=_TOKENS,
        snapshot_id=_SNAPSHOT,
        prior_token_id=_TOKENS[0],
        prior_action_id=0,
        prior_response_idx=1,
        behavior_llm_prior_logprob=prior_lp[0],
        prior_logits=prior,
        direct_all_action_q=direct_q,
        planner_root_mean_values=planner,
        planner_root_visit_counts=(4, 4),
        guided_action_id=1,
        behavior_guided_logprob=guided_lp[1],
        config=policy,
    )


def _row():
    behavior = _behavior()
    return {
        "group_idx": "g",
        "traj_idx": 0,
        "guided_turn_index": 0,
        "rollout_stop_reason": "task_failure",
        "decision_ledger": {
            "schema": "vagen_decision_ledger_v3_k4_mcts_guided",
            "action_space": "test_v1",
            "action_space_names": list(_ACTIONS),
            "executed_action_ids": [1],
            "executed_action_names": ["a1"],
            "decision_sources": ["k4_mcts_guided"],
            "decision_is_policy_sampled": [True],
            "env_turn_reward": 0.0,
            "env_terminated": False,
            "rollout_truncated": True,
            "format_valid": True,
            "snapshot_id": behavior.snapshot_id,
            "contract_id": behavior.contract_id,
            "behavior_record_id": behavior.record_id(),
            "behavior_record": behavior.to_mapping(),
        },
    }


class K4JointTrainingContractTest(unittest.TestCase):
    def test_training_contract_id_accepts_and_binds_k4_policy(self) -> None:
        from vagen.joint_policy.training_contract import joint_training_contract_id

        behavior = _behavior()
        first = joint_training_contract_id(
            _training_config(),
            behavior.policy_config,
            _wm_config(),
        )
        changed = type(behavior.policy_config).from_mapping(
            {**behavior.policy_config.to_mapping(), "beta": 3.0}
        )
        second = joint_training_contract_id(
            _training_config(), changed, _wm_config()
        )
        self.assertRegex(first, r"^sha256:[0-9a-f]{64}$")
        self.assertNotEqual(first, second)

    def test_frozen_v_uses_mcts_guided_policy_and_direct_q_values(self) -> None:
        from vagen.joint_policy.planning_contract import k4_guided_log_probs_reference
        from vagen.joint_policy.training_contract import (
            compile_outcome_returns_and_frozen_v_gae,
        )

        behavior = _behavior()
        guided = k4_guided_log_probs_reference(
            behavior.prior_logits,
            behavior.planner_root_mean_values,
            behavior.policy_config,
        )[1]
        expected = sum(
            math.exp(logp) * value
            for logp, value in zip(guided, behavior.direct_all_action_q, strict=True)
        )
        result = compile_outcome_returns_and_frozen_v_gae(
            [_row()],
            config=_training_config(),
        )
        self.assertTrue(math.isclose(result.frozen_state_values[0], expected))
        self.assertEqual(result.executed_action_ids, (1,))
        self.assertEqual(result.discounted_returns, (0.0,))

    def test_rejects_k4_ledger_source_or_schema_tampering(self) -> None:
        from vagen.joint_policy.training_contract import (
            compile_outcome_returns_and_frozen_v_gae,
        )

        row = _row()
        row["decision_ledger"]["decision_sources"] = ["frozen_q_guided"]
        with self.assertRaisesRegex(ValueError, "source"):
            compile_outcome_returns_and_frozen_v_gae(
                [row],
                config=_training_config(),
            )


if __name__ == "__main__":
    unittest.main()
