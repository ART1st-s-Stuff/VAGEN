from __future__ import annotations

import math
import unittest
from dataclasses import replace

from vagen.joint_policy import FrozenQGuidedPolicyConfig


_ACTION_NAMES = ("move_forward", "turn_right")
_ACTION_TOKEN_IDS = (101, 105)


def _config() -> FrozenQGuidedPolicyConfig:
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


def _sample(draw: float):
    from vagen.joint_policy import sample_frozen_q_guided_action

    return sample_frozen_q_guided_action(
        action_space="navigation_v1",
        action_space_names=_ACTION_NAMES,
        action_token_ids=_ACTION_TOKEN_IDS,
        prior_logits=(0.0, 0.0),
        frozen_all_action_q=(0.0, math.log(3.0)),
        uniform_draw=draw,
        config=_config(),
    )


class GuidedPolicySamplingTest(unittest.TestCase):
    def test_inverse_cdf_uses_half_open_intervals(self) -> None:
        from vagen.joint_policy import GUIDED_ACTION_DRAW_SCHEMA

        first = _sample(0.0)
        below = _sample(math.nextafter(0.25, 0.0))
        boundary = _sample(0.25)
        last = _sample(math.nextafter(1.0, 0.0))

        self.assertEqual(first.schema, GUIDED_ACTION_DRAW_SCHEMA)
        self.assertEqual(first.guided_action_id, 0)
        self.assertEqual(below.guided_action_id, 0)
        self.assertEqual(boundary.guided_action_id, 1)
        self.assertEqual(last.guided_action_id, 1)
        self.assertAlmostEqual(first.guided_log_probs[0], math.log(0.25))
        self.assertAlmostEqual(first.guided_log_probs[1], math.log(0.75))
        self.assertEqual(
            boundary.behavior_guided_logprob,
            boundary.guided_log_probs[1],
        )

    def test_common_nonbinary_cdf_boundary_uses_next_interval(self) -> None:
        from vagen.joint_policy import sample_frozen_q_guided_action

        record = sample_frozen_q_guided_action(
            action_space="navigation_v1",
            action_space_names=_ACTION_NAMES,
            action_token_ids=_ACTION_TOKEN_IDS,
            prior_logits=(math.log(0.7), math.log(0.3)),
            frozen_all_action_q=(0.0, 0.0),
            uniform_draw=0.7,
            config=_config(),
        )
        self.assertEqual(record.guided_action_id, 1)

    def test_same_external_draw_is_deterministic_and_zero_is_canonical(self) -> None:
        self.assertEqual(_sample(0.5), _sample(0.5))
        self.assertEqual(_sample(0.5).record_id(), _sample(0.5).record_id())
        positive = _sample(0.0)
        negative = _sample(-0.0)
        self.assertEqual(positive, negative)
        self.assertEqual(positive.record_id(), negative.record_id())
        self.assertEqual(math.copysign(1.0, negative.uniform_draw), 1.0)

    def test_signed_zero_in_persisted_vectors_has_one_audit_identity(self) -> None:
        from vagen.joint_policy import sample_frozen_q_guided_action

        positive = sample_frozen_q_guided_action(
            action_space="navigation_v1",
            action_space_names=_ACTION_NAMES,
            action_token_ids=_ACTION_TOKEN_IDS,
            prior_logits=(0.0, 0.0),
            frozen_all_action_q=(0.0, 0.0),
            uniform_draw=0.5,
            config=_config(),
        )
        negative = sample_frozen_q_guided_action(
            action_space="navigation_v1",
            action_space_names=_ACTION_NAMES,
            action_token_ids=_ACTION_TOKEN_IDS,
            prior_logits=(-0.0, 0.0),
            frozen_all_action_q=(-0.0, 0.0),
            uniform_draw=0.5,
            config=_config(),
        )
        self.assertEqual(positive, negative)
        self.assertEqual(positive.record_id(), negative.record_id())
        self.assertEqual(math.copysign(1.0, negative.prior_logits[0]), 1.0)
        self.assertEqual(math.copysign(1.0, negative.frozen_all_action_q[0]), 1.0)

    def test_signed_zero_beta_has_one_contract_and_record_identity(self) -> None:
        from vagen.joint_policy import sample_frozen_q_guided_action

        positive_config = _config()
        negative_config = FrozenQGuidedPolicyConfig.from_mapping(
            {**positive_config.__dict__, "beta": -0.0}
        )
        kwargs = {
            "action_space": "navigation_v1",
            "action_space_names": _ACTION_NAMES,
            "action_token_ids": _ACTION_TOKEN_IDS,
            "prior_logits": (0.0, 0.0),
            "frozen_all_action_q": (0.0, 0.0),
            "uniform_draw": 0.5,
        }
        positive = sample_frozen_q_guided_action(
            **kwargs,
            config=FrozenQGuidedPolicyConfig.from_mapping(
                {**positive_config.__dict__, "beta": 0.0}
            ),
        )
        negative = sample_frozen_q_guided_action(
            **kwargs,
            config=negative_config,
        )
        self.assertEqual(positive, negative)
        self.assertEqual(positive.contract_id, negative.contract_id)
        self.assertEqual(positive.record_id(), negative.record_id())

    def test_record_round_trip_revalidates_every_derived_field(self) -> None:
        from vagen.joint_policy import GuidedPolicyActionDrawRecord

        record = _sample(0.5)
        self.assertEqual(
            GuidedPolicyActionDrawRecord.from_mapping(record.to_mapping()),
            record,
        )
        cases = (
            ("schema", "future", "schema"),
            ("contract_id", "sha256:forged", "contract_id"),
            ("guided_action_id", 0, "guided_action_id"),
            ("behavior_guided_logprob", -123.0, "behavior_guided_logprob"),
            ("guided_log_probs", (0.0, 0.0), "guided_log_probs"),
        )
        for field, value, message in cases:
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                replace(record, **{field: value})

    def test_mapping_rejects_non_sequence_containers(self) -> None:
        from vagen.joint_policy import GuidedPolicyActionDrawRecord

        record = _sample(0.5)
        cases = (
            ("action_space_names", {"move_forward": 1, "turn_right": 2}),
            ("action_token_ids", {101: 1, 105: 2}),
            ("prior_logits", {0: 0.0, 1: 0.0}),
            ("frozen_all_action_q", {0: 0.0, 1: 1.0}),
            ("guided_log_probs", {0: -1.0, 1: -0.5}),
        )
        for field, value in cases:
            raw = record.to_mapping()
            raw[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                GuidedPolicyActionDrawRecord.from_mapping(raw)

    def test_contract_binds_config_action_table_and_token_ids(self) -> None:
        record = _sample(0.5)
        changed = record.to_mapping()
        changed["action_token_ids"] = [201, 205]
        with self.assertRaisesRegex(ValueError, "contract_id"):
            type(record).from_mapping(changed)

    def test_rejects_invalid_draw_without_coercion(self) -> None:
        for draw in (True, False, -0.1, 1.0, math.inf, -math.inf, math.nan, "0.5"):
            with self.subTest(draw=draw), self.assertRaisesRegex(
                ValueError,
                "uniform_draw",
            ):
                _sample(draw)  # type: ignore[arg-type]

    def test_rejects_shape_nonfinite_and_action_table_errors(self) -> None:
        from vagen.joint_policy import sample_frozen_q_guided_action

        base = {
            "action_space": "navigation_v1",
            "action_space_names": _ACTION_NAMES,
            "action_token_ids": _ACTION_TOKEN_IDS,
            "prior_logits": (0.0, 0.0),
            "frozen_all_action_q": (0.0, 0.0),
            "uniform_draw": 0.5,
            "config": _config(),
        }
        cases = (
            ({"prior_logits": (0.0,)}, "align"),
            ({"frozen_all_action_q": (0.0, float("nan"))}, "finite"),
            ({"action_space_names": ("move_forward",)}, "align"),
            ({"action_token_ids": (101, 101)}, "unique"),
        )
        for override, message in cases:
            with self.subTest(override=override), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                sample_frozen_q_guided_action(**{**base, **override})

    def test_extreme_distribution_still_selects_total_action_space(self) -> None:
        from vagen.joint_policy import sample_frozen_q_guided_action

        record = sample_frozen_q_guided_action(
            action_space="navigation_v1",
            action_space_names=_ACTION_NAMES,
            action_token_ids=_ACTION_TOKEN_IDS,
            prior_logits=(-1e9, 0.0),
            frozen_all_action_q=(0.0, 0.0),
            uniform_draw=0.0,
            config=_config(),
        )
        # Softmax is mathematically positive for action 0. The exact zero draw
        # belongs to the first half-open interval even if exp(logp) underflows.
        self.assertEqual(record.guided_action_id, 0)


if __name__ == "__main__":
    unittest.main()
