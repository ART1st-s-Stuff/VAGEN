from __future__ import annotations

import asyncio
import unittest


class _FailingEnv:
    def __init__(self) -> None:
        self.closed = False

    async def reset(self, _seed: int):
        raise RuntimeError("synthetic reset failure")

    async def close(self) -> None:
        self.closed = True


class NavigationHandlerCleanupTest(unittest.TestCase):
    def test_constructor_failure_releases_slot(self) -> None:
        try:
            from vagen.envs.navigation.handler import NavigationHandler
        except ImportError as exc:
            self.skipTest(f"navigation dependencies unavailable: {exc}")

        handler = NavigationHandler(devices=[0], max_envs=1)

        async def run() -> None:
            with self.assertRaisesRegex(ValueError, "requires explicit positive"):
                await handler.connect(
                    {
                        "eval_set": "base",
                        "prompt_format": "nimloth",
                        "max_actions_per_step": 1,
                    },
                    seed=None,
                )
            self.assertEqual(handler._active, {0: 0})
            await asyncio.wait_for(handler._env_slots.acquire(), timeout=0.1)
            handler._env_slots.release()
            await handler.aclose()

        asyncio.run(run())

    def test_initial_reset_failure_releases_session_and_slot(self) -> None:
        try:
            from vagen.envs.navigation.handler import NavigationHandler
        except ImportError as exc:
            self.skipTest(f"navigation dependencies unavailable: {exc}")

        handler = NavigationHandler(devices=[0], max_envs=1)
        env = _FailingEnv()

        async def acquire(_device, _config, _seed):
            await handler._env_slots.acquire()
            return env

        handler._acquire_env = acquire  # type: ignore[method-assign]

        async def run() -> None:
            with self.assertRaisesRegex(RuntimeError, "synthetic reset failure"):
                await handler.connect({"eval_set": "base"}, seed=0)
            self.assertEqual(handler._sessions, {})
            self.assertEqual(handler._active, {0: 0})
            self.assertTrue(env.closed)
            await asyncio.wait_for(handler._env_slots.acquire(), timeout=0.1)
            handler._env_slots.release()
            await handler.aclose()

        asyncio.run(run())

    def test_cache_reuse_requires_exact_environment_config(self) -> None:
        try:
            from vagen.envs.navigation.handler import NavigationHandler
        except ImportError as exc:
            self.skipTest(f"navigation dependencies unavailable: {exc}")

        handler = NavigationHandler(devices=[0], max_envs=2)

        class CachedEnv:
            config = {
                "eval_set": "base",
                "prompt_format": "free_think",
                "gpu_device": 0,
            }

        env = CachedEnv()
        handler._cache[0].append(("FloorPlan1", {"eval_set": "base", "prompt_format": "free_think"}, env))
        self.assertIsNone(
            handler._pop_cached(
                0,
                {
                    "eval_set": "base",
                    "prompt_format": "nimloth",
                    "latent_token_count": 16,
                    "gpu_device": 0,
                },
                "FloorPlan1",
            )
        )
        self.assertIs(
            handler._pop_cached(
                0,
                {
                    "eval_set": "base",
                    "prompt_format": "free_think",
                    "gpu_device": 0,
                },
                "FloorPlan1",
            ),
            env,
        )


if __name__ == "__main__":
    unittest.main()
