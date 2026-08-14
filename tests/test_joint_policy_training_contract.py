import math
import unittest

from vagen.joint_policy.contract import (
    FrozenQGuidedPolicyConfig,
    GuidedPolicyBehaviorRecord,
    guided_log_probs_reference,
)


_ACTIONS = tuple(f"a{index}" for index in range(3))
_TOKENS = (101, 102, 103)
_SNAPSHOT = "sha256:" + "1" * 64


def _policy() -> FrozenQGuidedPolicyConfig:
    return FrozenQGuidedPolicyConfig.from_mapping(
        {
            "implementation": "frozen_q_guided_v1",
            "alpha": 1.0,
            "beta": 1.0,
            "prior_temperature": 1.0,
            "backprop_to_llm": True,
            "score_dtype": "float32",
        }
    )


def _behavior(*, q=(1.0, 2.0, 3.0), action=0) -> dict:
    from vagen.joint_policy.contract import guided_log_probs_reference

    logits = (0.0, 0.0, 0.0)
    prior_log_probs, guided_log_probs = guided_log_probs_reference(
        logits,
        q,
        _policy(),
    )
    return GuidedPolicyBehaviorRecord.build(
        action_space="test_v1",
        action_space_names=_ACTIONS,
        action_token_ids=_TOKENS,
        snapshot_id=_SNAPSHOT,
        prior_token_id=_TOKENS[0],
        prior_action_id=0,
        prior_response_idx=4,
        behavior_llm_prior_logprob=prior_log_probs[0],
        prior_logits=logits,
        frozen_all_action_q=q,
        guided_action_id=action,
        behavior_guided_logprob=guided_log_probs[action],
        config=_policy(),
    ).to_mapping()


def _ledger(*, reward, terminated, truncated, behavior=None) -> dict:
    behavior = behavior or _behavior()
    return {
        "schema": "vagen_decision_ledger_v2_frozen_q_guided",
        "action_space": "test_v1",
        "action_space_names": list(_ACTIONS),
        "executed_action_ids": [behavior["guided_action_id"]],
        "executed_action_names": [_ACTIONS[behavior["guided_action_id"]]],
        "decision_sources": ["frozen_q_guided"],
        "decision_is_policy_sampled": [True],
        "env_turn_reward": reward,
        "env_terminated": terminated,
        "rollout_truncated": truncated,
        "format_valid": True,
        "contract_id": behavior["contract_id"],
        "snapshot_id": behavior["snapshot_id"],
        "behavior_record_id": GuidedPolicyBehaviorRecord.from_mapping(behavior).record_id(),
        "behavior_record": behavior,
    }


def _row(turn, *, stop_reason, reward=0.0, terminated=False, truncated=False, behavior=None):
    return {
        "group_idx": "group-1",
        "traj_idx": 0,
        "guided_turn_index": turn,
        "rollout_stop_reason": stop_reason,
        "decision_ledger": _ledger(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            behavior=behavior,
        ),
    }


def _config(**updates):
    raw = {
        "enabled": True,
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
        "critic_qwen_hidden_dim": 2048,
        "critic_grid_tokens": 16,
        "critic_state_dim": 1024,
        "critic_action_count": 3,
        "critic_huber_delta": 1.0,
        "critic_grad_clip": 2.0,
        "critic_optimizer": {
            "name": "adamw",
            "lr": 1e-4,
            "betas": [0.9, 0.95],
            "eps": 1e-8,
            "weight_decay": 0.01,
        },
    }
    raw.update(updates)
    return raw


class JointTrainingConfigTest(unittest.TestCase):
    def test_training_contract_id_binds_every_explicit_value(self) -> None:
        from vagen.joint_policy.contract import FrozenQGuidedPolicyConfig
        from vagen.joint_policy.training_contract import (
            joint_training_contract_id,
            parse_joint_training_section,
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
        first = parse_joint_training_section(_config())
        changed = _config()
        changed["actor_optimizer"] = dict(changed["actor_optimizer"])
        changed["actor_optimizer"]["lr"] = 2e-6
        second = parse_joint_training_section(changed)
        self.assertRegex(
            joint_training_contract_id(first, policy),
            r"^sha256:[0-9a-f]{64}$",
        )
        self.assertNotEqual(
            joint_training_contract_id(first, policy),
            joint_training_contract_id(second, policy),
        )

    def test_disabled_has_no_training_defaults(self) -> None:
        from vagen.joint_policy.training_contract import parse_joint_training_section

        self.assertIsNone(parse_joint_training_section({"enabled": False}))
        with self.assertRaisesRegex(ValueError, "unexpected"):
            parse_joint_training_section({"enabled": False, "gamma": 0.9})

    def test_enabled_requires_every_explicit_field(self) -> None:
        from vagen.joint_policy.training_contract import parse_joint_training_section

        config = parse_joint_training_section(_config())
        self.assertEqual(config.initial_snapshot_source_step, 776)
        self.assertEqual(config.critic_optimizer.betas, (0.9, 0.95))
        for field in tuple(_config()):
            raw = _config()
            raw.pop(field)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "missing"):
                parse_joint_training_section(raw)

    def test_rejects_invalid_algorithm_optimizer_and_dimensions(self) -> None:
        from vagen.joint_policy.training_contract import parse_joint_training_section

        bad_values = {
            "gamma": 0.0,
            "gae_lambda": 1.1,
            "ppo_clip_ratio": 1.0,
            "normalize_advantages": False,
            "token_kl_coefficient": -1.0,
            "token_kl_type": "forward_kl",
            "guided_entropy_coefficient": float("nan"),
            "initial_snapshot_source_step": True,
            "critic_grid_tokens": 0,
            "critic_huber_delta": 0.0,
            "critic_grad_clip": float("inf"),
        }
        for field, value in bad_values.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                parse_joint_training_section(_config(**{field: value}))
        with self.assertRaisesRegex(ValueError, "actor optimizer eps"):
            parse_joint_training_section(
                _config(
                    actor_optimizer={
                        **_config()["actor_optimizer"],
                        "eps": 0.0,
                    }
                )
            )
        with self.assertRaisesRegex(ValueError, "adamw"):
            parse_joint_training_section(
                _config(critic_optimizer={**_config()["critic_optimizer"], "name": "sgd"})
            )


class FrozenVGAETest(unittest.TestCase):
    def test_task_failure_has_zero_returns_and_frozen_v_gae(self) -> None:
        from vagen.joint_policy.training_contract import (
            compile_outcome_returns_and_frozen_v_gae,
            parse_joint_training_section,
        )

        rows = [
            _row(0, stop_reason="continue"),
            _row(1, stop_reason="task_failure", truncated=True),
        ]
        result = compile_outcome_returns_and_frozen_v_gae(
            rows,
            config=parse_joint_training_section(_config(normalize_advantages=True)),
        )
        self.assertEqual(result.discounted_returns, (0.0, 0.0))
        self.assertEqual(len(result.frozen_state_values), 2)
        expected_v = sum(
            math.exp(log_prob) * q
            for log_prob, q in zip(
                guided_log_probs_reference(
                    (0.0, 0.0, 0.0),
                    (1.0, 2.0, 3.0),
                    _policy(),
                )[1],
                (1.0, 2.0, 3.0),
                strict=True,
            )
        )
        self.assertTrue(all(math.isclose(v, expected_v) for v in result.frozen_state_values))
        self.assertTrue(math.isclose(sum(result.advantages), 0.0, abs_tol=1e-12))
        self.assertTrue(math.isclose(sum(a * a for a in result.advantages) / 2, 1.0, rel_tol=1e-6))

    def test_success_reward_is_discounted_for_selected_action_critic_target(self) -> None:
        from vagen.joint_policy.training_contract import (
            compile_outcome_returns_and_frozen_v_gae,
            parse_joint_training_section,
        )

        rows = [
            _row(0, stop_reason="continue"),
            _row(1, stop_reason="success", reward=10.0, terminated=True),
        ]
        result = compile_outcome_returns_and_frozen_v_gae(
            rows,
            config=parse_joint_training_section(_config()),
        )
        self.assertEqual(result.discounted_returns, (9.0, 10.0))
        self.assertEqual(result.executed_action_ids, (0, 0))

    def test_multiple_trajectories_restore_input_order_before_global_normalization(self) -> None:
        from vagen.joint_policy.training_contract import (
            compile_outcome_returns_and_frozen_v_gae,
            parse_joint_training_section,
        )

        rows = [
            {**_row(1, stop_reason="success", reward=5.0, terminated=True), "group_idx": "b"},
            {**_row(0, stop_reason="continue"), "group_idx": "a"},
            {**_row(0, stop_reason="continue"), "group_idx": "b"},
            {**_row(1, stop_reason="task_failure", truncated=True), "group_idx": "a"},
        ]
        result = compile_outcome_returns_and_frozen_v_gae(
            rows,
            config=parse_joint_training_section(_config()),
        )
        self.assertEqual(result.discounted_returns, (5.0, 0.0, 4.5, 0.0))
        self.assertEqual(len(result.advantages), 4)
        self.assertTrue(math.isclose(sum(result.advantages), 0.0, abs_tol=1e-9))

    def test_rejects_infrastructure_truncation_and_nonzero_failure_reward(self) -> None:
        from vagen.joint_policy.training_contract import (
            compile_outcome_returns_and_frozen_v_gae,
            parse_joint_training_section,
        )

        config = parse_joint_training_section(_config())
        with self.assertRaisesRegex(ValueError, "infrastructure"):
            compile_outcome_returns_and_frozen_v_gae(
                [_row(0, stop_reason="infrastructure_truncation", truncated=True)],
                config=config,
            )
        with self.assertRaisesRegex(ValueError, "failure.*zero"):
            compile_outcome_returns_and_frozen_v_gae(
                [_row(0, stop_reason="task_failure", reward=1.0, truncated=True)],
                config=config,
            )

    def test_rejects_action_dependent_q_baseline_and_malformed_sequences(self) -> None:
        from vagen.joint_policy.training_contract import (
            compile_outcome_returns_and_frozen_v_gae,
            parse_joint_training_section,
        )

        config = parse_joint_training_section(_config())
        malformed = [
            _row(0, stop_reason="continue"),
            _row(2, stop_reason="task_failure", truncated=True),
        ]
        with self.assertRaisesRegex(ValueError, "contiguous"):
            compile_outcome_returns_and_frozen_v_gae(malformed, config=config)
        with self.assertRaisesRegex(TypeError, "selected-action Q baseline"):
            compile_outcome_returns_and_frozen_v_gae(
                [_row(0, stop_reason="task_failure", truncated=True)],
                config=config,
                selected_action_q_baseline=True,
            )


if __name__ == "__main__":
    unittest.main()
