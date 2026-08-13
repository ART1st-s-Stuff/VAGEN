from __future__ import annotations

import math
import unittest
from dataclasses import replace

from vagen.joint_policy import FrozenQGuidedPolicyConfig


_ACTION_NAMES = ("move_forward", "turn_right")
_ACTION_TOKEN_IDS = (101, 105)
_SNAPSHOT_ID = "sha256:" + "1" * 64


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


def _key(
    sample_id: str,
    *,
    config: FrozenQGuidedPolicyConfig | None = None,
    action_names=_ACTION_NAMES,
    action_token_ids=_ACTION_TOKEN_IDS,
):
    from vagen.joint_policy import GuidedActionDrawKey

    config = config or _config()
    return GuidedActionDrawKey.build(
        run_seed=1,
        policy_step=1,
        rollout_sample_id=sample_id,
        rollout_repeat_index=0,
        turn_index=0,
        is_validation=False,
        snapshot_id=_SNAPSHOT_ID,
        contract_id=config.contract_id(
            "navigation_v1",
            action_names,
            action_token_ids,
        ),
    )


def _sample(sample_id: str = "sample-0"):
    from vagen.joint_policy import sample_frozen_q_guided_action

    return sample_frozen_q_guided_action(
        action_space="navigation_v1",
        action_space_names=_ACTION_NAMES,
        action_token_ids=_ACTION_TOKEN_IDS,
        prior_logits=(0.0, 0.0),
        frozen_all_action_q=(0.0, math.log(3.0)),
        draw_key=_key(sample_id),
        config=_config(),
    )


class GuidedPolicySamplingTest(unittest.TestCase):
    def test_inverse_cdf_uses_half_open_intervals(self) -> None:
        from vagen.joint_policy.sampling import _inverse_cdf_action

        log_probs = (math.log(0.25), math.log(0.75))
        self.assertEqual(_inverse_cdf_action(log_probs, 0.0), 0)
        self.assertEqual(_inverse_cdf_action(log_probs, math.nextafter(0.25, 0.0)), 0)
        self.assertEqual(_inverse_cdf_action(log_probs, 0.25), 1)
        self.assertEqual(_inverse_cdf_action(log_probs, math.nextafter(1.0, 0.0)), 1)

    def test_keyed_records_select_from_the_derived_draw(self) -> None:
        from vagen.joint_policy import GUIDED_ACTION_DRAW_SCHEMA

        first = _sample("sample-1")  # derived draw 0.1969... < 0.25
        second = _sample("sample-0")  # derived draw 0.9917... >= 0.25
        self.assertEqual(first.schema, GUIDED_ACTION_DRAW_SCHEMA)
        self.assertEqual(first.guided_action_id, 0)
        self.assertEqual(second.guided_action_id, 1)
        self.assertAlmostEqual(first.guided_log_probs[0], math.log(0.25))
        self.assertAlmostEqual(first.guided_log_probs[1], math.log(0.75))
        self.assertEqual(
            second.behavior_guided_logprob,
            second.guided_log_probs[1],
        )

    def test_common_nonbinary_cdf_boundary_uses_next_interval(self) -> None:
        from vagen.joint_policy.sampling import _inverse_cdf_action

        self.assertEqual(
            _inverse_cdf_action((math.log(0.7), math.log(0.3)), 0.7),
            1,
        )

    def test_same_key_is_deterministic(self) -> None:
        self.assertEqual(_sample("sample-0"), _sample("sample-0"))
        self.assertEqual(
            _sample("sample-0").record_id(),
            _sample("sample-0").record_id(),
        )

    def test_signed_zero_in_persisted_vectors_has_one_audit_identity(self) -> None:
        from vagen.joint_policy import sample_frozen_q_guided_action

        key = _key("sample-zero")
        positive = sample_frozen_q_guided_action(
            action_space="navigation_v1",
            action_space_names=_ACTION_NAMES,
            action_token_ids=_ACTION_TOKEN_IDS,
            prior_logits=(0.0, 0.0),
            frozen_all_action_q=(0.0, 0.0),
            draw_key=key,
            config=_config(),
        )
        negative = sample_frozen_q_guided_action(
            action_space="navigation_v1",
            action_space_names=_ACTION_NAMES,
            action_token_ids=_ACTION_TOKEN_IDS,
            prior_logits=(-0.0, 0.0),
            frozen_all_action_q=(-0.0, 0.0),
            draw_key=key,
            config=_config(),
        )
        self.assertEqual(positive, negative)
        self.assertEqual(positive.record_id(), negative.record_id())
        self.assertEqual(math.copysign(1.0, negative.prior_logits[0]), 1.0)
        self.assertEqual(math.copysign(1.0, negative.frozen_all_action_q[0]), 1.0)

    def test_signed_zero_beta_has_one_contract_and_record_identity(self) -> None:
        from vagen.joint_policy import sample_frozen_q_guided_action

        positive_config = FrozenQGuidedPolicyConfig.from_mapping(
            {**_config().__dict__, "beta": 0.0}
        )
        negative_config = FrozenQGuidedPolicyConfig.from_mapping(
            {**_config().__dict__, "beta": -0.0}
        )
        kwargs = {
            "action_space": "navigation_v1",
            "action_space_names": _ACTION_NAMES,
            "action_token_ids": _ACTION_TOKEN_IDS,
            "prior_logits": (0.0, 0.0),
            "frozen_all_action_q": (0.0, 0.0),
        }
        positive = sample_frozen_q_guided_action(
            **kwargs,
            draw_key=_key("sample-beta", config=positive_config),
            config=positive_config,
        )
        negative = sample_frozen_q_guided_action(
            **kwargs,
            draw_key=_key("sample-beta", config=negative_config),
            config=negative_config,
        )
        self.assertEqual(positive, negative)
        self.assertEqual(positive.contract_id, negative.contract_id)
        self.assertEqual(positive.record_id(), negative.record_id())

    def test_record_round_trip_revalidates_every_derived_field(self) -> None:
        from vagen.joint_policy import GuidedPolicyActionDrawRecord

        record = _sample()
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
            ("uniform_draw", 0.5, "uniform_draw"),
        )
        for field, value, message in cases:
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                replace(record, **{field: value})

    def test_mapping_rejects_non_sequence_containers(self) -> None:
        from vagen.joint_policy import GuidedPolicyActionDrawRecord

        record = _sample()
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

    def test_contract_binds_config_action_table_token_ids_and_key(self) -> None:
        record = _sample()
        changed = record.to_mapping()
        changed["action_token_ids"] = [201, 205]
        with self.assertRaisesRegex(ValueError, "contract_id"):
            type(record).from_mapping(changed)

        changed = record.to_mapping()
        changed["draw_key"]["contract_id"] = "sha256:" + "3" * 64
        with self.assertRaisesRegex(ValueError, "contract_id"):
            type(record).from_mapping(changed)

    def test_rejects_shape_nonfinite_and_action_table_errors(self) -> None:
        from vagen.joint_policy import sample_frozen_q_guided_action

        base = {
            "action_space": "navigation_v1",
            "action_space_names": _ACTION_NAMES,
            "action_token_ids": _ACTION_TOKEN_IDS,
            "prior_logits": (0.0, 0.0),
            "frozen_all_action_q": (0.0, 0.0),
            "draw_key": _key("sample-errors"),
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

    def test_extreme_distribution_inverse_cdf_covers_first_action_at_zero(self) -> None:
        from vagen.joint_policy.sampling import _inverse_cdf_action

        # Softmax is mathematically positive for action 0. The exact zero draw
        # belongs to the first half-open interval even if exp(logp) underflows.
        self.assertEqual(_inverse_cdf_action((-1e9, 0.0), 0.0), 0)


if __name__ == "__main__":
    unittest.main()
