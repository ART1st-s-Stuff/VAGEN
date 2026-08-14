import unittest


class JointPolicyOutcomeTest(unittest.TestCase):
    def test_classifies_task_outcomes_and_infrastructure_stop(self) -> None:
        from vagen.joint_policy.outcome import classify_rollout_stop_reason

        self.assertEqual(
            classify_rollout_stop_reason(
                success=True,
                env_terminated=True,
                turn_count=2,
                max_turns=4,
                response_limit_exhausted=False,
            ),
            "success",
        )
        self.assertEqual(
            classify_rollout_stop_reason(
                success=False,
                env_terminated=True,
                turn_count=2,
                max_turns=4,
                response_limit_exhausted=False,
            ),
            "environment_failure",
        )
        self.assertEqual(
            classify_rollout_stop_reason(
                success=False,
                env_terminated=False,
                turn_count=4,
                max_turns=4,
                response_limit_exhausted=False,
            ),
            "task_failure",
        )
        self.assertEqual(
            classify_rollout_stop_reason(
                success=False,
                env_terminated=False,
                turn_count=1,
                max_turns=4,
                response_limit_exhausted=True,
            ),
            "infrastructure_truncation",
        )
        self.assertEqual(
            classify_rollout_stop_reason(
                success=False,
                env_terminated=False,
                turn_count=1,
                max_turns=4,
                response_limit_exhausted=False,
            ),
            "continue",
        )

    def test_rejects_success_without_environment_terminal_and_bad_types(self) -> None:
        from vagen.joint_policy.outcome import classify_rollout_stop_reason

        with self.assertRaisesRegex(ValueError, "success.*terminal"):
            classify_rollout_stop_reason(
                success=True,
                env_terminated=False,
                turn_count=1,
                max_turns=4,
                response_limit_exhausted=False,
            )
        with self.assertRaises(ValueError):
            classify_rollout_stop_reason(
                success=1,
                env_terminated=False,
                turn_count=1,
                max_turns=4,
                response_limit_exhausted=False,
            )


if __name__ == "__main__":
    unittest.main()
