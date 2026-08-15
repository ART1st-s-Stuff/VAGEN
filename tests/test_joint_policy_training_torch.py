import unittest


class JointPolicyTrainingTorchTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"torch unavailable: {exc}")

    def _policy(self):
        from vagen.joint_policy.contract import FrozenQGuidedPolicyConfig

        return FrozenQGuidedPolicyConfig.from_mapping(
            {
                "implementation": "frozen_q_guided_v1",
                "alpha": 1.0,
                "beta": 0.5,
                "prior_temperature": 1.0,
                "backprop_to_llm": True,
                "score_dtype": "float32",
            }
        )

    def test_guided_action_ppo_uses_actual_action_and_stops_q_gradient(self) -> None:
        import torch

        from vagen.joint_policy.training_torch import guided_action_ppo_terms

        logits = torch.tensor(
            [[0.2, -0.1, 0.3], [0.1, 0.4, -0.2]],
            dtype=torch.float32,
            requires_grad=True,
        )
        frozen_q = torch.tensor(
            [[1.0, 0.0, -1.0], [0.5, -0.5, 0.25]],
            dtype=torch.float32,
            requires_grad=True,
        )
        actions = torch.tensor([2, 0], dtype=torch.long)
        behavior = torch.tensor([-1.0, -1.2], dtype=torch.float32)
        advantages = torch.tensor([1.5, -0.5], dtype=torch.float32)
        terms = guided_action_ppo_terms(
            current_prior_logits=logits,
            frozen_all_action_q=frozen_q,
            guided_action_ids=actions,
            behavior_guided_log_probs=behavior,
            advantages=advantages,
            valid_mask=torch.tensor([True, True]),
            policy_config=self._policy(),
            clip_ratio=0.2,
        )
        loss = terms.policy_loss_sum - 0.01 * terms.entropy_sum
        loss.backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertIsNone(frozen_q.grad)
        self.assertEqual(tuple(terms.selected_current_log_probs.shape), (2,))
        self.assertTrue(torch.isfinite(terms.ratios).all())

    def test_k4_guided_action_ppo_uses_mcts_and_stops_planner_gradient(self) -> None:
        import torch

        from vagen.joint_policy.planning_contract import K4MCTSGuidedPolicyConfig
        from vagen.joint_policy.training_torch import k4_guided_action_ppo_terms

        policy = K4MCTSGuidedPolicyConfig.from_mapping(
            {
                "implementation": "k4_mcts_guided_v1",
                "alpha": 1.0,
                "beta": 2.0,
                "prior_temperature": 1.0,
                "backprop_to_llm": True,
                "score_dtype": "float32",
                "planning_horizon": 4,
                "mcts_num_simulations": 100,
                "mcts_exploration_constant": 1.0,
            }
        )
        logits = torch.tensor(
            [[0.2, -0.1], [0.1, 0.4]], requires_grad=True
        )
        planner = torch.tensor(
            [[-0.5, 0.5], [0.25, -0.25]], requires_grad=True
        )
        actions = torch.tensor([1, 0], dtype=torch.long)
        with torch.no_grad():
            guided = torch.log_softmax(logits + 2.0 * planner, dim=-1)
            behavior = guided.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
        terms = k4_guided_action_ppo_terms(
            current_prior_logits=logits,
            frozen_planner_root_mean_values=planner,
            guided_action_ids=actions,
            behavior_guided_log_probs=behavior,
            advantages=torch.tensor([1.0, -1.0]),
            valid_mask=torch.tensor([True, True]),
            policy_config=policy,
            clip_ratio=0.2,
        )
        (terms.policy_loss_sum - 0.01 * terms.entropy_sum).backward()
        self.assertIsNotNone(logits.grad)
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)
        self.assertIsNone(planner.grad)
        self.assertTrue(torch.allclose(terms.ratios, torch.ones(2)))

    def test_clipping_and_padding_mask_are_applied_per_executed_turn(self) -> None:
        import torch

        from vagen.joint_policy.training_torch import guided_action_ppo_terms

        logits = torch.tensor(
            [[10.0, 0.0], [0.0, 10.0]],
            dtype=torch.float32,
            requires_grad=True,
        )
        q_values = torch.zeros_like(logits)
        actions = torch.tensor([0, 1], dtype=torch.long)
        behavior = torch.tensor([-5.0, -5.0], dtype=torch.float32)
        advantages = torch.tensor([2.0, 999.0], dtype=torch.float32)
        terms = guided_action_ppo_terms(
            current_prior_logits=logits,
            frozen_all_action_q=q_values,
            guided_action_ids=actions,
            behavior_guided_log_probs=behavior,
            advantages=advantages,
            valid_mask=torch.tensor([1, 0]),
            policy_config=self._policy(),
            clip_ratio=0.2,
        )
        self.assertEqual(int(terms.valid_count), 1)
        self.assertTrue(torch.isclose(terms.clipped_ratios[0], torch.tensor(1.2)))
        self.assertTrue(torch.isclose(terms.policy_loss_sum, torch.tensor(-2.4)))

    def test_low_variance_token_kl_uses_policy_tokens_and_reference_stopgrad(self) -> None:
        import torch

        from vagen.joint_policy.training_torch import low_variance_token_kl_terms

        current = torch.tensor(
            [[-1.0, -2.0, -3.0], [-0.5, -0.7, -0.9]],
            dtype=torch.float32,
            requires_grad=True,
        )
        reference = torch.tensor(
            [[-1.1, -2.2, -3.3], [-0.4, -0.6, -0.8]],
            dtype=torch.float32,
            requires_grad=True,
        )
        terms = low_variance_token_kl_terms(
            current_token_log_probs=current,
            reference_token_log_probs=reference,
            response_mask=torch.tensor([[1, 0, 1], [1, 1, 1]]),
            valid_row_mask=torch.tensor([True, False]),
        )
        self.assertEqual(int(terms.valid_token_count), 2)
        self.assertGreaterEqual(float(terms.kl_sum), 0.0)
        terms.kl_sum.backward()
        self.assertIsNotNone(current.grad)
        self.assertIsNone(reference.grad)
        self.assertEqual(float(current.grad[0, 1]), 0.0)
        self.assertEqual(float(current.grad[1].abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
