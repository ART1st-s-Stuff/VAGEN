from __future__ import annotations

import inspect
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


def _key(**overrides):
    from vagen.joint_policy import GuidedActionDrawKey

    values = {
        "run_seed": 1234,
        "policy_step": 7,
        "rollout_sample_id": "train-record-42",
        "rollout_repeat_index": 3,
        "turn_index": 5,
        "is_validation": False,
        "snapshot_id": _SNAPSHOT_ID,
        "contract_id": _config().contract_id(
            "navigation_v1",
            _ACTION_NAMES,
            _ACTION_TOKEN_IDS,
        ),
    }
    values.update(overrides)
    return GuidedActionDrawKey.build(**values)


class GuidedActionDrawKeyTest(unittest.TestCase):
    def test_keyed_draw_has_a_stable_golden_identity_and_value(self) -> None:
        from vagen.joint_policy import (
            GUIDED_ACTION_DRAW_KEY_SCHEMA,
            GuidedActionDrawKey,
        )

        key = _key()
        self.assertEqual(key.schema, GUIDED_ACTION_DRAW_KEY_SCHEMA)
        self.assertEqual(
            key.key_id(),
            "sha256:cab93f25d809a5e4518e203c1a00a0da4872f3431da44d921cbb8b5b915b314d",
        )
        self.assertEqual(key.uniform_draw(), 0.7918891398804022)
        self.assertEqual(GuidedActionDrawKey.from_mapping(key.to_mapping()), key)

    def test_same_logical_decision_is_stateless_and_schedule_independent(self) -> None:
        first = _key()
        second = _key()
        self.assertEqual(first, second)
        self.assertEqual(first.key_id(), second.key_id())
        self.assertEqual(first.uniform_draw(), second.uniform_draw())
        self.assertNotIn("worker", first.to_mapping())
        self.assertNotIn("request_id", first.to_mapping())
        self.assertNotIn("generation_id", first.to_mapping())

    def test_every_logical_identity_field_is_hash_bound(self) -> None:
        baseline = _key()
        changes = (
            {"run_seed": 1235},
            {"policy_step": 8},
            {"rollout_sample_id": "train-record-43"},
            {"rollout_repeat_index": 4},
            {"turn_index": 6},
            {"is_validation": True},
            {"snapshot_id": "sha256:" + "2" * 64},
            {"contract_id": "sha256:" + "3" * 64},
        )
        for override in changes:
            with self.subTest(override=override):
                changed = _key(**override)
                self.assertNotEqual(changed.key_id(), baseline.key_id())

    def test_key_direct_and_mapping_construction_fail_closed(self) -> None:
        from vagen.joint_policy import GuidedActionDrawKey

        key = _key()
        cases = (
            ("run_seed", True),
            ("run_seed", -1),
            ("policy_step", -1),
            ("rollout_sample_id", ""),
            ("rollout_repeat_index", True),
            ("turn_index", -1),
            ("is_validation", 0),
            ("snapshot_id", ""),
            ("contract_id", ""),
        )
        for field, value in cases:
            with self.subTest(field=field), self.assertRaises(ValueError):
                replace(key, **{field: value})

        missing = key.to_mapping()
        missing.pop("turn_index")
        with self.assertRaisesRegex(ValueError, "missing fields"):
            GuidedActionDrawKey.from_mapping(missing)
        unexpected = key.to_mapping()
        unexpected["worker_id"] = "worker-9"
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            GuidedActionDrawKey.from_mapping(unexpected)

    def test_action_record_derives_draw_and_persists_full_provenance(self) -> None:
        from vagen.joint_policy import (
            GUIDED_ACTION_DRAW_SCHEMA,
            GuidedPolicyActionDrawRecord,
            sample_frozen_q_guided_action,
        )

        key = _key()
        record = sample_frozen_q_guided_action(
            action_space="navigation_v1",
            action_space_names=_ACTION_NAMES,
            action_token_ids=_ACTION_TOKEN_IDS,
            prior_logits=(0.0, 0.0),
            frozen_all_action_q=(0.0, math.log(3.0)),
            draw_key=key,
            config=_config(),
        )
        self.assertEqual(record.schema, GUIDED_ACTION_DRAW_SCHEMA)
        self.assertEqual(record.draw_key, key)
        self.assertEqual(record.draw_key.key_id(), key.key_id())
        self.assertEqual(record.uniform_draw, key.uniform_draw())
        self.assertEqual(
            GuidedPolicyActionDrawRecord.from_mapping(record.to_mapping()),
            record,
        )
        self.assertIn("draw_key", record.to_mapping())

        with self.assertRaisesRegex(ValueError, "uniform_draw"):
            replace(record, uniform_draw=math.nextafter(record.uniform_draw, 1.0))

    def test_sampler_has_no_caller_selected_draw_or_rng_state(self) -> None:
        from vagen.joint_policy import sample_frozen_q_guided_action
        from vagen.joint_policy import sampling

        signature = inspect.signature(sample_frozen_q_guided_action)
        self.assertIn("draw_key", signature.parameters)
        self.assertNotIn("uniform_draw", signature.parameters)
        source = inspect.getsource(sampling)
        self.assertNotIn("import random", source)
        self.assertNotIn("import secrets", source)
        self.assertNotIn("numpy", source)

    def test_draw_key_contract_must_match_distribution_contract(self) -> None:
        from vagen.joint_policy import sample_frozen_q_guided_action

        with self.assertRaisesRegex(ValueError, "contract"):
            sample_frozen_q_guided_action(
                action_space="navigation_v1",
                action_space_names=_ACTION_NAMES,
                action_token_ids=_ACTION_TOKEN_IDS,
                prior_logits=(0.0, 0.0),
                frozen_all_action_q=(0.0, math.log(3.0)),
                draw_key=_key(contract_id="sha256:" + "3" * 64),
                config=_config(),
            )


if __name__ == "__main__":
    unittest.main()
