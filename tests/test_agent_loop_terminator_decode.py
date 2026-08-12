from __future__ import annotations

import unittest


class _Tokenizer:
    eos_token_id = 151645
    pad_token_id = 151643


class AgentLoopTerminatorDecodeTest(unittest.TestCase):
    def test_strips_only_trailing_generation_terminators(self) -> None:
        try:
            from vagen.agent_loop.gym_agent_loop_no_concat import (
                _strip_trailing_generation_terminators,
            )
        except ImportError as exc:
            self.skipTest(f"agent-loop dependencies unavailable: {exc}")

        action_end_id = 151690
        response_ids = [10, action_end_id, 151645, 151643]
        original = list(response_ids)

        self.assertEqual(
            _strip_trailing_generation_terminators(response_ids, _Tokenizer()),
            [10, action_end_id],
        )
        self.assertEqual(response_ids, original)

    def test_does_not_strip_terminator_like_interior_tokens(self) -> None:
        try:
            from vagen.agent_loop.gym_agent_loop_no_concat import (
                _strip_trailing_generation_terminators,
            )
        except ImportError as exc:
            self.skipTest(f"agent-loop dependencies unavailable: {exc}")

        self.assertEqual(
            _strip_trailing_generation_terminators(
                [10, 151645, 11],
                _Tokenizer(),
            ),
            [10, 151645, 11],
        )


if __name__ == "__main__":
    unittest.main()
