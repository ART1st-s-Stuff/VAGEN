from __future__ import annotations

import unittest


class _Tokenizer:
    all_special_ids = [90, 100, 101, 102, 103, 104]
    added_tokens_decoder = {}

    def encode(self, text, add_special_tokens=False):
        assert text == "</think>"
        assert add_special_tokens is False
        return [7, 8]

    def convert_tokens_to_ids(self, token):
        token_ids = {
            "<|latent_state|>": 90,
            "<|latent_state_1|>": 91,
            "<|action_start|>": 92,
            "<|action_end|>": 93,
            "<|action_(0)|>": 100,
            "<|action_(1)|>": 101,
            "<|action_(2)|>": 102,
            "<|action_(3)|>": 103,
            "<|action_(4)|>": 104,
            "<|action_(5)|>": 105,
            "<|action_(6)|>": 106,
            "<|action_(7)|>": 107,
        }
        return token_ids[token]

    def encode_token(self, token):
        return [self.convert_tokens_to_ids(token)]

    def __call__(self, token, add_special_tokens=False):
        return {"input_ids": self.encode_token(token)}


class NimlothTurnGenerationWiringTest(unittest.TestCase):
    def test_sample_mask_excludes_forced_protocol_tokens(self) -> None:
        try:
            from vagen.agent_loop.gym_agent_loop_no_concat import (
                _nimloth_response_mask,
            )
            from nimloth.backbone.qwen25vl.turn_generation import TurnGenerationSpec
        except ImportError as exc:
            self.skipTest(f"Nimloth/VAGEN dependencies unavailable: {exc}")

        spec = TurnGenerationSpec(
            close_text="</think>",
            close_token_ids=(7, 8),
            injected_token_ids=(90, 91, 92),
            action_token_ids=(100, 101),
            action_end_token_id=93,
            forbidden_reasoning_token_ids=(),
            max_reasoning_tokens=4,
        )
        response = [1, 2, 7, 8, 90, 91, 92, 101, 93]
        self.assertEqual(
            _nimloth_response_mask(response, spec),
            [1, 1, 1, 1, 0, 0, 0, 1, 0],
        )

    def test_prompt_prefills_open_think_tag(self) -> None:
        from pathlib import Path

        source = (
            Path(__file__).resolve().parents[1]
            / "vagen/agent_loop/gym_agent_loop_no_concat.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'messages.append({"role": "assistant", "content": "<think>"})',
            source,
        )
        self.assertIn('"continue_final_message": True', source)
        self.assertIn('f"<think>{assistant_message}"', source)

    def test_invalid_or_missing_protocol_fails_closed(self) -> None:
        try:
            from vagen.agent_loop.gym_agent_loop_no_concat import (
                _nimloth_response_mask,
            )
            from nimloth.backbone.qwen25vl.turn_generation import TurnGenerationSpec
        except ImportError as exc:
            self.skipTest(f"Nimloth/VAGEN dependencies unavailable: {exc}")

        spec = TurnGenerationSpec(
            close_text="</think>",
            close_token_ids=(7, 8),
            injected_token_ids=(90, 91, 92),
            action_token_ids=(100, 101),
            action_end_token_id=93,
            forbidden_reasoning_token_ids=(),
            max_reasoning_tokens=4,
        )
        for response in ([1, 2, 7, 8, 100, 93], [7, 8, 90, 91, 92, 99, 93]):
            with self.subTest(response=response), self.assertRaises(RuntimeError):
                _nimloth_response_mask(response, spec)


if __name__ == "__main__":
    unittest.main()
