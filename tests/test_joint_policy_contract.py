from __future__ import annotations

import math
import unittest

from vagen.joint_policy.contract import (
    GUIDED_BEHAVIOR_SCHEMA,
    FrozenQGuidedPolicyConfig,
    GuidedPolicyBehaviorRecord,
    guided_log_probs_reference,
    parse_joint_policy_section,
)


_ACTION_SPACE = "navigation_v1"
_ACTION_NAMES = ["move_forward", "turn_right"]
_ACTION_TOKEN_IDS = [101, 105]


def _config(*, backprop_to_llm: bool = True, alpha: float = 1.0, beta: float = 1.0):
    return FrozenQGuidedPolicyConfig.from_mapping(
        {
            "implementation": "frozen_q_guided_v1",
            "alpha": alpha,
            "beta": beta,
            "prior_temperature": 1.0,
            "backprop_to_llm": backprop_to_llm,
            "score_dtype": "float64",
        }
    )


def _record(**overrides):
    kwargs = {
        "action_space": _ACTION_SPACE,
        "action_space_names": _ACTION_NAMES,
        "action_token_ids": _ACTION_TOKEN_IDS,
        "snapshot_id": "critic-step-0007",
        "prior_token_id": 101,
        "prior_action_id": 0,
        "prior_response_idx": 12,
        "behavior_llm_prior_logprob": -0.4,
        "prior_logits": [0.0, 0.0],
        "frozen_all_action_q": [0.0, math.log(3.0)],
        "guided_action_id": 1,
        "behavior_guided_logprob": math.log(0.75),
        "config": _config(),
    }
    kwargs.update(overrides)
    return GuidedPolicyBehaviorRecord.build(**kwargs)


class FrozenQGuidedPolicyConfigTest(unittest.TestCase):
    def test_requires_gradient_semantic_explicitly(self) -> None:
        raw = {
            "implementation": "frozen_q_guided_v1",
            "alpha": 1.0,
            "beta": 0.25,
            "prior_temperature": 1.0,
            "score_dtype": "float64",
        }
        with self.assertRaisesRegex(ValueError, "backprop_to_llm"):
            FrozenQGuidedPolicyConfig.from_mapping(raw)

    def test_disabled_section_does_not_invent_policy_defaults(self) -> None:
        self.assertIsNone(parse_joint_policy_section({"enabled": False}))

    def test_enabled_section_requires_llm_gradient_path(self) -> None:
        for invalid_choice in (None, False):
            with self.subTest(backprop_to_llm=invalid_choice):
                with self.assertRaisesRegex(ValueError, "must be true"):
                    parse_joint_policy_section(
                        {
                            "enabled": True,
                            "implementation": "frozen_q_guided_v1",
                            "alpha": 1.0,
                            "beta": 0.25,
                            "prior_temperature": 1.0,
                            "backprop_to_llm": invalid_choice,
                            "score_dtype": "float64",
                        }
                    )

    def test_requires_positive_prior_weight_for_llm_gradient(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha must be positive"):
            _config(alpha=0.0, beta=1.0)

    def test_direct_dataclass_construction_cannot_bypass_invariants(self) -> None:
        with self.assertRaisesRegex(ValueError, "alpha must be positive"):
            FrozenQGuidedPolicyConfig(
                implementation="frozen_q_guided_v1",
                alpha=0.0,
                beta=1.0,
                prior_temperature=1.0,
                backprop_to_llm=True,
                score_dtype="float32",
            )
        with self.assertRaisesRegex(ValueError, "must be true"):
            FrozenQGuidedPolicyConfig(
                implementation="frozen_q_guided_v1",
                alpha=1.0,
                beta=1.0,
                prior_temperature=1.0,
                backprop_to_llm=False,
                score_dtype="float32",
            )

    def test_contract_id_binds_config_action_table_and_tokens(self) -> None:
        config = _config(backprop_to_llm=True)
        changed_alpha = _config(backprop_to_llm=True, alpha=0.5)
        contract_id = config.contract_id(
            _ACTION_SPACE,
            _ACTION_NAMES,
            _ACTION_TOKEN_IDS,
        )
        self.assertEqual(
            contract_id,
            config.contract_id(_ACTION_SPACE, _ACTION_NAMES, _ACTION_TOKEN_IDS),
        )
        self.assertNotEqual(
            contract_id,
            changed_alpha.contract_id(_ACTION_SPACE, _ACTION_NAMES, _ACTION_TOKEN_IDS),
        )
        self.assertNotEqual(
            contract_id,
            config.contract_id(_ACTION_SPACE, _ACTION_NAMES, [201, 205]),
        )


class GuidedPolicyMathTest(unittest.TestCase):
    def test_combines_scaled_prior_and_frozen_q(self) -> None:
        prior_log_probs, guided_log_probs = guided_log_probs_reference(
            prior_logits=[0.0, 0.0],
            frozen_q=[0.0, math.log(3.0)],
            config=_config(),
        )
        self.assertAlmostEqual(prior_log_probs[0], math.log(0.5))
        self.assertAlmostEqual(prior_log_probs[1], math.log(0.5))
        self.assertAlmostEqual(guided_log_probs[0], math.log(0.25))
        self.assertAlmostEqual(guided_log_probs[1], math.log(0.75))

    def test_rejects_overflow_after_scaling(self) -> None:
        with self.assertRaisesRegex(ValueError, "guided_logits must be finite"):
            guided_log_probs_reference(
                prior_logits=[1e308, -1e308],
                frozen_q=[0.0, 0.0],
                config=_config(alpha=1e308, beta=0.0),
            )


class GuidedPolicyBehaviorRecordTest(unittest.TestCase):
    def test_builds_versioned_auditable_behavior(self) -> None:
        record = _record()
        self.assertEqual(record.schema, GUIDED_BEHAVIOR_SCHEMA)
        self.assertEqual(record.guided_action_id, 1)
        self.assertEqual(record.prior_token_id, _ACTION_TOKEN_IDS[0])
        self.assertTrue(record.contract_id.startswith("sha256:"))
        self.assertAlmostEqual(record.behavior_guided_logprob, math.log(0.75))

    def test_rejects_prior_token_not_matching_action_mapping(self) -> None:
        with self.assertRaisesRegex(ValueError, "prior token id"):
            _record(prior_token_id=999)

    def test_rejects_positive_llm_logprob(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be positive"):
            _record(behavior_llm_prior_logprob=0.1)

    def test_logprob_comparison_uses_absolute_tolerance(self) -> None:
        config = _config(alpha=1.0, beta=1.0)
        _, guided = guided_log_probs_reference(
            [-1e9, 0.0],
            [0.0, 0.0],
            config,
        )
        with self.assertRaisesRegex(ValueError, "behavior guided log-prob"):
            _record(
                prior_logits=[-1e9, 0.0],
                frozen_all_action_q=[0.0, 0.0],
                guided_action_id=0,
                behavior_guided_logprob=guided[0] + 999.0,
                config=config,
            )

    def test_strict_mapping_round_trip_and_unknown_schema_rejection(self) -> None:
        record = _record()
        restored = GuidedPolicyBehaviorRecord.from_mapping(record.to_mapping())
        self.assertEqual(restored, record)
        invalid = record.to_mapping()
        invalid["schema"] = "future"
        with self.assertRaisesRegex(ValueError, "unsupported guided behavior schema"):
            GuidedPolicyBehaviorRecord.from_mapping(invalid)


if __name__ == "__main__":
    unittest.main()
