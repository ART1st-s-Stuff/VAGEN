from __future__ import annotations

import math
import unittest
from dataclasses import replace

from vagen.agent_loop.decision_ledger import (
    GUIDED_DECISION_LEDGER_SCHEMA,
    build_guided_decision_ledger,
    validate_decision_ledger,
)
from vagen.joint_policy import FrozenQGuidedPolicyConfig, GuidedPolicyBehaviorRecord


_ACTION_NAMES = ["move_forward", "turn_right"]
_ACTION_TOKEN_IDS = [101, 105]


def _config():
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


def _behavior():
    return GuidedPolicyBehaviorRecord.build(
        action_space="navigation_v1",
        action_space_names=_ACTION_NAMES,
        action_token_ids=_ACTION_TOKEN_IDS,
        snapshot_id="critic-step-0007",
        prior_token_id=101,
        prior_action_id=0,
        prior_response_idx=12,
        behavior_llm_prior_logprob=-0.4,
        prior_logits=[0.0, 0.0],
        frozen_all_action_q=[0.0, math.log(3.0)],
        guided_action_id=1,
        behavior_guided_logprob=math.log(0.75),
        config=_config(),
    )


class GuidedDecisionLedgerTest(unittest.TestCase):
    def test_derives_policy_ownership_from_validated_behavior(self) -> None:
        behavior = _behavior()
        ledger = build_guided_decision_ledger(
            behavior=behavior,
            env_turn_reward=1.0,
            env_terminated=False,
            rollout_truncated=False,
            format_valid=True,
        )
        self.assertEqual(ledger["schema"], GUIDED_DECISION_LEDGER_SCHEMA)
        self.assertEqual(ledger["executed_action_ids"], [1])
        self.assertEqual(ledger["executed_action_names"], ["turn_right"])
        self.assertEqual(ledger["decision_sources"], ["frozen_q_guided"])
        self.assertEqual(ledger["decision_is_policy_sampled"], [True])
        self.assertEqual(ledger["snapshot_id"], behavior.snapshot_id)
        self.assertEqual(ledger["contract_id"], behavior.contract_id)
        self.assertEqual(ledger["behavior_record_id"], behavior.record_id())
        validate_decision_ledger(ledger)

    def test_directly_forged_behavior_object_is_revalidated(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported guided behavior schema"):
            build_guided_decision_ledger(
                behavior=replace(_behavior(), schema="forged"),
                env_turn_reward=0.0,
                env_terminated=False,
                rollout_truncated=False,
                format_valid=True,
            )

    def test_unknown_action_id_cannot_be_forged_at_ledger_boundary(self) -> None:
        behavior = _behavior()
        invalid = behavior.to_mapping()
        invalid["guided_action_id"] = 2
        with self.assertRaisesRegex(ValueError, "outside action space"):
            GuidedPolicyBehaviorRecord.from_mapping(invalid)


if __name__ == "__main__":
    unittest.main()
