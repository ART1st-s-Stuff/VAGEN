from __future__ import annotations

import unittest


class NavigationConfigContractTest(unittest.TestCase):
    def test_nimloth_requires_explicit_latent_count(self) -> None:
        try:
            from vagen.envs.navigation.navigation_env import NavigationEnv
        except ImportError as exc:
            self.skipTest(f"navigation dependencies unavailable: {exc}")

        with self.assertRaisesRegex(ValueError, "requires explicit positive"):
            NavigationEnv(
                {
                    "eval_set": "base",
                    "prompt_format": "nimloth",
                    "max_actions_per_step": 1,
                }
            )

        env = NavigationEnv(
            {
                "eval_set": "base",
                "prompt_format": "nimloth",
                "max_actions_per_step": 1,
                "latent_token_count": 16,
            }
        )
        self.assertEqual(env.cfg.latent_token_count, 16)


if __name__ == "__main__":
    unittest.main()
