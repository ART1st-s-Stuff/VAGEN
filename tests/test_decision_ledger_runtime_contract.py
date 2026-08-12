from __future__ import annotations

import unittest

from vagen.agent_loop.decision_ledger import (
    DECISION_LEDGER_SCHEMA,
    GUIDED_DECISION_LEDGER_SCHEMA,
    build_decision_ledger,
    parse_decision_ledger_enabled,
    summarize_decision_ledger_batch,
    validate_decision_ledger_reward_rows,
)


_ACTION_NAMES = ["move_forward", "turn_right"]


def _ledger(*, reward: float = 0.5):
    return build_decision_ledger(
        action_space="navigation_v1",
        action_space_names=_ACTION_NAMES,
        executed_action_ids=[0],
        executed_action_names=["move_forward"],
        decision_source="llm_text",
        env_turn_reward=reward,
        env_terminated=False,
        rollout_truncated=False,
        format_valid=True,
    )


class DecisionLedgerConfigTest(unittest.TestCase):
    def test_only_literal_bool_is_accepted(self) -> None:
        self.assertFalse(parse_decision_ledger_enabled(None))
        self.assertFalse(parse_decision_ledger_enabled({"enabled": False}))
        self.assertTrue(parse_decision_ledger_enabled({"enabled": True}))
        for invalid in ("false", "true", 0, 1, None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "explicit bool"):
                    parse_decision_ledger_enabled({"enabled": invalid})

    def test_unknown_fields_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            parse_decision_ledger_enabled({"enabled": False, "future": 1})


class DecisionLedgerTrainerBoundaryTest(unittest.TestCase):
    def test_stock_trainer_can_restrict_batch_to_v1(self) -> None:
        ledger = _ledger()
        metrics = summarize_decision_ledger_batch(
            [ledger],
            expected_batch_size=1,
            allowed_schemas={DECISION_LEDGER_SCHEMA},
        )
        self.assertEqual(metrics["decision_ledger/turn_coverage"], 1.0)

        guided = dict(ledger)
        guided["schema"] = GUIDED_DECISION_LEDGER_SCHEMA
        with self.assertRaisesRegex(ValueError, "not allowed"):
            summarize_decision_ledger_batch(
                [guided],
                expected_batch_size=1,
                allowed_schemas={DECISION_LEDGER_SCHEMA},
            )

    def test_reward_rows_match_ledger_and_last_policy_token(self) -> None:
        validate_decision_ledger_reward_rows(
            [_ledger(reward=0.5), _ledger(reward=-0.25)],
            reward_rows=[[0.0, 0.5, 0.0], [0.0, 0.0, -0.25]],
            response_masks=[[1, 1, 0], [1, 1, 1]],
        )

    def test_reward_mismatch_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match ledger"):
            validate_decision_ledger_reward_rows(
                [_ledger(reward=0.5)],
                reward_rows=[[0.0, 0.25]],
                response_masks=[[1, 1]],
            )

    def test_reward_on_non_anchor_token_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside the last policy-owned token"):
            validate_decision_ledger_reward_rows(
                [_ledger(reward=0.5)],
                reward_rows=[[0.5, 0.0]],
                response_masks=[[1, 1]],
            )


if __name__ == "__main__":
    unittest.main()
