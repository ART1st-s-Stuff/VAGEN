from __future__ import annotations

import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


def _source(relative_path: str) -> str:
    return (_ROOT / relative_path).read_text(encoding="utf-8")


class DecisionLedgerWiringTest(unittest.TestCase):
    def test_agent_loop_closes_environment_on_ledger_failure(self) -> None:
        source = _source("vagen/agent_loop/gym_agent_loop_no_concat.py")
        method_start = source.index("async def run(")
        method_end = source.index("async def _handle_pending_state", method_start)
        method = source[method_start:method_end]
        self.assertIn("try:", method)
        self.assertIn("finally:\n            await env.close()", method)

    def test_upstream_action_text_reaches_environment_unchanged(self) -> None:
        source = _source("vagen/agent_loop/gym_agent_loop_no_concat.py")
        method_start = source.index("async def _handle_env_state")
        step_call = source.index("await agent_data.env.step(action_str)", method_start)
        method_prefix = source[method_start:step_call]
        self.assertIn('action_str = agent_data.last_assistant_text or ""', method_prefix)
        self.assertNotIn("_adapt_action_for_latent_planner", method_prefix)
        self.assertNotIn("injected_suffix", method_prefix)

    def test_turn_reward_anchors_before_injected_suffix(self) -> None:
        source = _source("vagen/agent_loop/agent_loop_no_concat.py")
        self.assertIn(
            "last_policy_token_index(mask.tolist()) for mask in response_mask",
            source,
        )
        self.assertNotIn(
            "response_length = attention_mask[:, prompt_length:].sum(dim=1) - 1",
            source,
        )

    def test_scheme_b_tensor_path_always_detaches_q(self) -> None:
        source = _source("vagen/joint_policy/torch_policy.py")
        self.assertIn("q_guidance = frozen_all_action_q.detach()", source)
        self.assertIn("policy_prior_logits = prior_logits", source)
        self.assertNotIn("prior_logits.detach()", source)
        self.assertIn("config.alpha * scaled_prior_logits + config.beta * q_guidance", source)

    def test_joint_policy_config_is_validated_before_workers_start(self) -> None:
        source = _source("vagen/ray_trainer.py")
        parser = source.index("self.joint_policy_config = parse_joint_policy_section(")
        dataloader = source.index("self._create_dataloader(")
        self.assertLess(parser, dataloader)
        self.assertIn(
            '"joint_policy.enabled requires decision_ledger.enabled=true"',
            source,
        )
        self.assertIn(
            '"and replay are not connected; refusing to run stock PPO"',
            source,
        )

    def test_navigation_exports_versioned_action_space_and_executed_actions(self) -> None:
        source = _source("vagen/envs/navigation/navigation_env.py")
        self.assertIn('"action_space": "navigation_v1"', source)
        self.assertIn('"action_space_names": list(ACTION_LOOKUP)', source)
        self.assertIn('"executed_action_names": list(self._valid_actions)', source)
        self.assertIn('"executed_action_ids": [ACTION_LOOKUP[action] - 1', source)
        self.assertIn('"planner_fallback_used": False', source)

    def test_extra_fields_reach_dataproto_without_first_action_reduction(self) -> None:
        source = _source("vagen/agent_loop/agent_loop_no_concat.py")
        self.assertIn(
            "all_keys = set(key for input_item in inputs for key in input_item.extra_fields)",
            source,
        )
        self.assertIn("non_tensor_batch.update(extra_fields)", source)

    def test_final_reward_tensor_is_revalidated_before_ppo_scores(self) -> None:
        source = _source("vagen/ray_trainer.py")
        final_validation = source.rindex("validate_decision_ledger_reward_rows(")
        score_assignment = source.index(
            'batch.batch["token_level_scores"] = reward_tensor',
            final_validation,
        )
        self.assertLess(final_validation, score_assignment)

    def test_validation_runs_before_old_log_prob_replay(self) -> None:
        source = _source("vagen/ray_trainer.py")
        validation_call = source.index(
            "self._validate_decision_ledger_batch(batch, metrics)"
        )
        replay_call = source.index("old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)")
        self.assertLess(validation_call, replay_call)

    def test_feature_rejects_concat_or_synchronous_rollout(self) -> None:
        source = _source("vagen/ray_trainer.py")
        self.assertIn(
            '"decision_ledger.enabled requires async rollout with "',
            source,
        )
        self.assertIn('self.config.actor_rollout_ref.rollout.mode != "async"', source)
        self.assertIn('self.config.trainer.get("concat_multi_turn", True)', source)


if __name__ == "__main__":
    unittest.main()
