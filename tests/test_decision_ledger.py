from __future__ import annotations

import unittest

from vagen.agent_loop.decision_ledger import (
    DECISION_LEDGER_SCHEMA,
    build_decision_ledger,
    build_decision_ledger_from_env_info,
    last_policy_token_index,
    summarize_decision_ledger_batch,
    validate_decision_ledger,
)


_ACTION_SPACE = "navigation_v1"
_ACTION_SPACE_NAMES = [
    "move_forward",
    "move_backward",
    "move_right",
    "move_left",
    "turn_right",
    "turn_left",
    "look_up",
    "look_down",
]


def _build_ledger(
    *,
    action_ids: list[int] | None = None,
    action_names: list[str] | None = None,
    source: str = "llm_text",
    reward: float = 0.0,
    terminated: bool = False,
    truncated: bool = False,
    format_valid: bool = True,
):
    return build_decision_ledger(
        action_space=_ACTION_SPACE,
        action_space_names=_ACTION_SPACE_NAMES,
        executed_action_ids=[] if action_ids is None else action_ids,
        executed_action_names=[] if action_names is None else action_names,
        decision_source=source,
        env_turn_reward=reward,
        env_terminated=terminated,
        rollout_truncated=truncated,
        format_valid=format_valid,
    )


class BuildDecisionLedgerTest(unittest.TestCase):
    def test_preserves_every_executed_action_in_order(self) -> None:
        ledger = _build_ledger(
            action_ids=[0, 4, 6],
            action_names=["move_forward", "turn_right", "look_up"],
            reward=0.25,
        )

        self.assertEqual(ledger["schema"], DECISION_LEDGER_SCHEMA)
        self.assertEqual(ledger["action_space"], _ACTION_SPACE)
        self.assertEqual(ledger["action_space_names"], _ACTION_SPACE_NAMES)
        self.assertEqual(ledger["executed_action_ids"], [0, 4, 6])
        self.assertEqual(
            ledger["executed_action_names"],
            ["move_forward", "turn_right", "look_up"],
        )
        self.assertEqual(ledger["decision_sources"], ["llm_text"] * 3)
        self.assertEqual(ledger["decision_is_policy_sampled"], [False] * 3)
        self.assertEqual(ledger["env_turn_reward"], 0.25)

    def test_system_fallback_never_claims_policy_sampling(self) -> None:
        ledger = _build_ledger(
            action_ids=[0],
            action_names=["move_forward"],
            source="system_fallback",
            terminated=True,
        )

        self.assertEqual(ledger["decision_sources"], ["system_fallback"])
        self.assertEqual(ledger["decision_is_policy_sampled"], [False])
        validate_decision_ledger(ledger)

    def test_action_ids_and_names_must_align(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            _build_ledger(
                action_ids=[0, 4],
                action_names=["move_forward"],
            )

    def test_action_id_must_match_canonical_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match action space"):
            _build_ledger(action_ids=[0], action_names=["look_down"])

    def test_invalid_executed_action_id_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "invalid executed action id"):
            _build_ledger(action_ids=[-1], action_names=["move_forward"])

    def test_boolean_reward_is_not_coerced_to_float(self) -> None:
        with self.assertRaisesRegex(ValueError, "env_turn_reward must be finite"):
            _build_ledger(reward=True)  # type: ignore[arg-type]

    def test_environment_contract_is_explicit(self) -> None:
        ledger = build_decision_ledger_from_env_info(
            {
                "action_space": _ACTION_SPACE,
                "action_space_names": _ACTION_SPACE_NAMES,
                "executed_action_ids": [0, 4],
                "executed_action_names": ["move_forward", "turn_right"],
                "format_correct": True,
                "planner_fallback_used": False,
            },
            env_turn_reward=0.5,
            env_terminated=False,
            rollout_truncated=False,
        )
        self.assertEqual(ledger["executed_action_ids"], [0, 4])
        self.assertEqual(ledger["decision_sources"], ["llm_text", "llm_text"])
        self.assertTrue(ledger["format_valid"])

    def test_environment_contract_rejects_missing_ids(self) -> None:
        with self.assertRaisesRegex(ValueError, "executed_action_ids"):
            build_decision_ledger_from_env_info(
                {
                    "action_space": _ACTION_SPACE,
                    "action_space_names": _ACTION_SPACE_NAMES,
                    "executed_action_names": ["move_forward"],
                    "format_correct": True,
                    "planner_fallback_used": False,
                },
                env_turn_reward=0.0,
                env_terminated=False,
                rollout_truncated=False,
            )

    def test_environment_contract_rejects_string_booleans(self) -> None:
        with self.assertRaisesRegex(ValueError, "format_correct must be bool"):
            build_decision_ledger_from_env_info(
                {
                    "action_space": _ACTION_SPACE,
                    "action_space_names": _ACTION_SPACE_NAMES,
                    "executed_action_ids": [0],
                    "executed_action_names": ["move_forward"],
                    "format_correct": "false",
                    "planner_fallback_used": False,
                },
                env_turn_reward=0.0,
                env_terminated=False,
                rollout_truncated=False,
            )


class TokenOwnershipTest(unittest.TestCase):
    def test_reward_anchor_skips_system_injected_suffix(self) -> None:
        self.assertEqual(last_policy_token_index([1, 1, 1, 0, 0]), 2)

    def test_reward_anchor_rejects_turn_without_policy_token(self) -> None:
        with self.assertRaisesRegex(ValueError, "no policy-owned token"):
            last_policy_token_index([0, 0])


class ValidateDecisionLedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.ledger = _build_ledger(
            action_ids=[0, 4],
            action_names=["move_forward", "turn_right"],
            reward=1.0,
            truncated=True,
        )

    def test_rejects_mismatched_action_field_lengths(self) -> None:
        self.ledger["decision_sources"] = ["llm_text"]
        with self.assertRaisesRegex(ValueError, "same length"):
            validate_decision_ledger(self.ledger)

    def test_m1_rejects_unimplemented_policy_ownership(self) -> None:
        self.ledger["decision_is_policy_sampled"] = [True, False]
        with self.assertRaisesRegex(ValueError, "does not define actor-policy sampling"):
            validate_decision_ledger(self.ledger)

    def test_rejects_terminal_and_truncated_turn(self) -> None:
        self.ledger["env_terminated"] = True
        with self.assertRaisesRegex(ValueError, "both terminated and truncated"):
            validate_decision_ledger(self.ledger)

    def test_rejects_unknown_schema(self) -> None:
        self.ledger["schema"] = "future_schema"
        with self.assertRaisesRegex(ValueError, "unsupported decision ledger schema"):
            validate_decision_ledger(self.ledger)


class SummarizeDecisionLedgerBatchTest(unittest.TestCase):
    def test_reports_turn_action_and_ownership_coverage(self) -> None:
        ledgers = [
            _build_ledger(
                action_ids=[0, 4],
                action_names=["move_forward", "turn_right"],
                reward=0.5,
            ),
            _build_ledger(
                action_ids=[0],
                action_names=["move_forward"],
                source="system_fallback",
                reward=1.0,
                terminated=True,
            ),
            _build_ledger(truncated=True, format_valid=False),
        ]

        metrics = summarize_decision_ledger_batch(ledgers, expected_batch_size=3)

        self.assertEqual(metrics["decision_ledger/turn_coverage"], 1.0)
        self.assertEqual(metrics["decision_ledger/executed_action_count_mean"], 1.0)
        self.assertEqual(metrics["decision_ledger/action_turn_coverage"], 2 / 3)
        self.assertEqual(metrics["decision_ledger/system_fallback_action_fraction"], 1 / 3)
        self.assertEqual(metrics["decision_ledger/policy_sampled_action_fraction"], 0.0)
        self.assertEqual(metrics["decision_ledger/terminated_turn_fraction"], 1 / 3)
        self.assertEqual(metrics["decision_ledger/truncated_turn_fraction"], 1 / 3)
        self.assertEqual(metrics["decision_ledger/format_valid_turn_fraction"], 2 / 3)

    def test_missing_ledger_fails_strict_batch_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "expected 2 decision ledgers"):
            summarize_decision_ledger_batch([{}], expected_batch_size=2)


if __name__ == "__main__":
    unittest.main()
