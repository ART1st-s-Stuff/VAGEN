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


if __name__ == "__main__":
    unittest.main()
