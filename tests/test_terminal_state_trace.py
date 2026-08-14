from pathlib import Path
import unittest


class TerminalStateTraceTest(unittest.TestCase):
    def test_terminal_generation_requests_latent_only_capture(self) -> None:
        source = (
            Path(__file__).parents[1]
            / "vagen/agent_loop/gym_agent_loop_no_concat.py"
        ).read_text()
        self.assertIn(
            '"nimloth_policy_state_capture_mode": "terminal_latent_only"',
            source,
        )
        self.assertIn('"nimloth_terminal_latent_state_v1"', source)

    def _kwargs(self):
        return {
            "request_id": "request-1",
            "generation_id": "generation-1",
            "rollout_stop_reason": "task_failure",
            "raw_response": "<think>real final thought</think><|latent_state|>",
            "response_ids": [11, 12, 21, 22],
            "response_mask": [1, 1, 0, 0],
            "response_logprobs": [-0.1, -0.2, 0.0, 0.0],
            "latent_token_ids": [21],
            "action_start_token_id": 22,
            "latent_hidden": [[0.5, -0.25]],
        }

    def test_round_trip_binds_real_generation_without_action(self) -> None:
        from vagen.joint_policy.terminal_state import TerminalStateTrace

        trace = TerminalStateTrace.build(**self._kwargs())
        self.assertEqual(trace.rollout_stop_reason, "task_failure")
        self.assertEqual(trace.response_ids[-1], trace.action_start_token_id)
        self.assertNotIn("action_token_ids", trace.to_mapping())
        self.assertNotIn("action_logits", trace.to_mapping())
        self.assertEqual(TerminalStateTrace.from_mapping(trace.to_mapping()), trace)
        self.assertTrue(trace.record_id().startswith("sha256:"))

    def test_rejects_template_thought_action_suffix_and_bad_identity(self) -> None:
        from vagen.joint_policy.terminal_state import TerminalStateTrace

        for updates, pattern in (
            ({"request_id": "generation-1"}, "differ"),
            ({"raw_response": "<think></think>"}, "real CoT"),
            (
                {
                    "response_ids": [11, 12, 21, 22, 99],
                    "response_mask": [1, 1, 0, 0, 1],
                    "response_logprobs": [-0.1, -0.2, 0.0, 0.0, -0.3],
                },
                "action_start",
            ),
            ({"response_mask": [1, 1, 0, 1]}, "forced"),
            ({"rollout_stop_reason": "continue"}, "outcome"),
        ):
            raw = self._kwargs()
            raw.update(updates)
            with self.subTest(updates=updates), self.assertRaisesRegex(ValueError, pattern):
                TerminalStateTrace.build(**raw)

    def test_revalidates_hidden_shape_finite_values_and_record_id(self) -> None:
        from vagen.joint_policy.terminal_state import TerminalStateTrace

        trace = TerminalStateTrace.build(**self._kwargs())
        raw = trace.to_mapping()
        raw["latent_hidden"] = [[float("nan"), 0.0]]
        with self.assertRaisesRegex(ValueError, "finite"):
            TerminalStateTrace.from_mapping(raw)
        raw = trace.to_mapping()
        raw["record_id"] = "sha256:forged"
        with self.assertRaisesRegex(ValueError, "unexpected"):
            TerminalStateTrace.from_mapping(raw)


if __name__ == "__main__":
    unittest.main()
