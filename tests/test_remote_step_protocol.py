from __future__ import annotations

import math
import unittest

from vagen.utils.remote_step_protocol import parse_remote_step_fields


class ParseRemoteStepFieldsTest(unittest.TestCase):
    def test_accepts_finite_reward_boolean_done_and_mapping_info(self) -> None:
        reward, done, info = parse_remote_step_fields(
            {"reward": 0.25, "done": False, "info": {"format_correct": True}}
        )
        self.assertEqual(reward, 0.25)
        self.assertFalse(done)
        self.assertEqual(info, {"format_correct": True})

    def test_rejects_boolean_reward(self) -> None:
        with self.assertRaisesRegex(ValueError, "reward must be a finite number"):
            parse_remote_step_fields({"reward": True, "done": False, "info": {}})

    def test_rejects_non_finite_reward(self) -> None:
        for reward in (math.inf, -math.inf, math.nan):
            with self.subTest(reward=reward):
                with self.assertRaisesRegex(ValueError, "reward must be a finite number"):
                    parse_remote_step_fields({"reward": reward, "done": False, "info": {}})

    def test_rejects_string_done(self) -> None:
        with self.assertRaisesRegex(ValueError, "done must be bool"):
            parse_remote_step_fields({"reward": 0.0, "done": "false", "info": {}})

    def test_rejects_missing_protocol_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing reward"):
            parse_remote_step_fields({"done": False, "info": {}})
        with self.assertRaisesRegex(ValueError, "missing done"):
            parse_remote_step_fields({"reward": 0.0, "info": {}})

    def test_rejects_non_mapping_info(self) -> None:
        with self.assertRaisesRegex(ValueError, "info must be a mapping"):
            parse_remote_step_fields({"reward": 0.0, "done": False, "info": []})


if __name__ == "__main__":
    unittest.main()
