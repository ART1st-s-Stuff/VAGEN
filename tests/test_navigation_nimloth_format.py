from __future__ import annotations

import unittest
from pathlib import Path

from vagen.envs.navigation.utils.nimloth_format import (
    ACTION_NAMES,
    latent_state_tokens,
)
from vagen.envs.navigation.utils.parse import parse_response
from vagen.envs.navigation.utils.prompt import get_format_instruction, system_prompt


_ROOT = Path(__file__).resolve().parents[1]


class NimlothNavigationFormatTest(unittest.TestCase):
    def test_checked_in_rollout_configs_select_k16(self) -> None:
        for relative in (
            "examples/train/navigation/train_navigation_nimloth_k16.yaml",
            "examples/train/navigation/val_navigation_nimloth_k16.yaml",
        ):
            with self.subTest(relative=relative):
                text = (_ROOT / relative).read_text(encoding="utf-8")
                self.assertIn("prompt_format: nimloth", text)
                self.assertIn("latent_token_count: 16", text)
                self.assertIn("max_actions_per_step: 1", text)

    def test_action_token_ids_follow_navigation_action_order(self) -> None:
        self.assertEqual(
            ACTION_NAMES,
            (
                "move_forward",
                "move_backward",
                "move_right",
                "move_left",
                "turn_right",
                "turn_left",
                "look_up",
                "look_down",
            ),
        )

    def test_complete_system_prompt_has_only_single_action_guidance(self) -> None:
        prompt = system_prompt(
            "nimloth",
            max_actions_per_step=1,
            example_count=0,
            latent_token_count=16,
        )
        self.assertIn("Choose exactly one valid action", prompt)
        self.assertNotIn("take multiple actions", prompt)
        self.assertNotIn("<action>", prompt)
        with self.assertRaisesRegex(ValueError, "example_count=0"):
            system_prompt(
                "nimloth",
                max_actions_per_step=1,
                example_count=1,
                latent_token_count=16,
            )

    def test_prompt_expands_exact_k16_latent_block(self) -> None:
        prompt = get_format_instruction(
            "nimloth",
            max_actions_per_step=1,
            action_sep="|",
            latent_token_count=16,
        )
        latent_block = "".join(latent_state_tokens(16))
        self.assertIn(
            latent_block + "<|action_start|><|action_(idx)|><|action_end|>",
            prompt,
        )

    def test_nimloth_parser_preserves_one_action_mapping(self) -> None:
        response = (
            "<think>Turn left.</think>"
            + "".join(latent_state_tokens(16))
            + "<|action_start|><|action_(5)|><|action_end|>"
        )
        parsed = parse_response(
            response,
            prompt_format="nimloth",
            max_actions=1,
            latent_token_count=16,
        )
        self.assertTrue(parsed["format_correct"])
        self.assertEqual(parsed["actions"], ["turn_left"])

    def test_parser_rejects_missing_or_misordered_latent_block(self) -> None:
        responses = (
            "<think>x</think><|action_start|><|action_(0)|><|action_end|>",
            (
                "<think>x</think><|action_start|>"
                + "".join(latent_state_tokens(16))
                + "<|action_(0)|><|action_end|>"
            ),
        )
        for response in responses:
            with self.subTest(response=response):
                parsed = parse_response(
                    response,
                    prompt_format="nimloth",
                    max_actions=1,
                    latent_token_count=16,
                )
                self.assertFalse(parsed["format_correct"])
                self.assertEqual(parsed["actions"], [])

    def test_parser_rejects_unknown_duplicate_or_extra_action_tokens(self) -> None:
        latent = "".join(latent_state_tokens(16))
        blocks = (
            "<|action_(9)|>",
            "<|action_(0)|><|action_(1)|>",
            "<|action_(0)|>junk",
        )
        for block in blocks:
            with self.subTest(block=block):
                response = (
                    f"<think>x</think>{latent}<|action_start|>"
                    f"{block}<|action_end|>"
                )
                parsed = parse_response(
                    response,
                    prompt_format="nimloth",
                    max_actions=1,
                    latent_token_count=16,
                )
                self.assertFalse(parsed["format_correct"])
                self.assertEqual(parsed["actions"], [])

    def test_parser_rejects_extra_or_unknown_latent_control_tokens(self) -> None:
        valid = (
            "<think>x</think>"
            + "".join(latent_state_tokens(16))
            + "<|action_start|><|action_(0)|><|action_end|>"
        )
        for response in (
            valid + "<|latent_state_7|>",
            valid + "<|latent_state_16|>",
            valid + "<|latent_state_x|>",
            valid + "<|action_(x)|>",
            valid + "<prediction>old wm tail</prediction>",
            "<|latent_state_16|>" + valid,
        ):
            with self.subTest(response=response):
                parsed = parse_response(
                    response,
                    prompt_format="nimloth",
                    max_actions=1,
                    latent_token_count=16,
                )
                self.assertFalse(parsed["format_correct"])
                self.assertEqual(parsed["actions"], [])

    def test_nimloth_requires_single_action_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one action"):
            get_format_instruction(
                "nimloth",
                max_actions_per_step=2,
                action_sep="|",
                latent_token_count=16,
            )


if __name__ == "__main__":
    unittest.main()
