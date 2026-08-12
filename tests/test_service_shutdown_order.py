from __future__ import annotations

import asyncio
import unittest


class ServiceShutdownOrderTest(unittest.TestCase):
    def test_handler_closes_before_executor_callback(self) -> None:
        try:
            from vagen.envs_remote.service import GymService
        except ImportError as exc:
            self.skipTest(f"service dependencies unavailable: {exc}")

        events: list[str] = []

        class Handler:
            async def aclose(self) -> None:
                events.append("handler")

        service = GymService(Handler())  # type: ignore[arg-type]
        app = service.build(shutdown_callback=lambda: events.append("executor"))

        async def run() -> None:
            async with app.router.lifespan_context(app):
                events.append("running")

        asyncio.run(run())
        self.assertEqual(events, ["running", "handler", "executor"])


if __name__ == "__main__":
    unittest.main()
