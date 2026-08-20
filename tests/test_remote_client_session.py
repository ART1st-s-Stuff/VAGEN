from __future__ import annotations

import asyncio
import unittest
from unittest import mock


class _Response:
    def __init__(self, *, status_code: int = 200) -> None:
        self.status_code = status_code
        self.headers = {"content-type": "application/octet-stream"}
        self.content = b"payload"

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _Client:
    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []

    async def post(self, url, **_kwargs):
        self.urls.append(url)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class RemoteSessionRoutingTest(unittest.TestCase):
    def test_scoped_transport_override_preserves_dataset_config(self) -> None:
        try:
            from vagen.envs_remote import gym_image_env_client as module
        except ImportError as exc:
            self.skipTest(f"remote-client dependencies unavailable: {exc}")

        config = {
            "base_urls": "http://immutable-checkpoint-url",
            "timeout": 10,
            "eval_set": "base_train",
        }
        with mock.patch.dict(
            module.os.environ,
            {
                "VAGEN_REMOTE_ENV_BASE_URL_OVERRIDE": "http://current-server",
                "VAGEN_REMOTE_ENV_BASE_URL_OVERRIDE_SCOPE": (
                    "id186_exact_continuation_v1"
                ),
            },
            clear=False,
        ):
            env = module.GymImageEnvClient(config)
        self.assertEqual(env.base_urls, ["http://current-server"])
        self.assertEqual(config["base_urls"], "http://immutable-checkpoint-url")
        self.assertEqual(env._remote_env_config, {"eval_set": "base_train"})

    def test_transport_override_fails_without_exact_scope(self) -> None:
        try:
            from vagen.envs_remote import gym_image_env_client as module
        except ImportError as exc:
            self.skipTest(f"remote-client dependencies unavailable: {exc}")

        with mock.patch.dict(
            module.os.environ,
            {"VAGEN_REMOTE_ENV_BASE_URL_OVERRIDE": "http://current-server"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "exact ID186"):
                module.GymImageEnvClient(
                    {"base_urls": "http://immutable-checkpoint-url"}
                )

    def test_state_mutating_step_is_never_retried(self) -> None:
        try:
            from vagen.envs_remote import gym_image_env_client as module
        except ImportError as exc:
            self.skipTest(f"remote-client dependencies unavailable: {exc}")

        env = module.GymImageEnvClient(
            {
                "base_urls": ["http://server-a"],
                "retries": 3,
                "backoff": 0.0,
                "log_retries": False,
            }
        )
        env._session_id = "session-on-a"
        env._client = _Client([RuntimeError("ambiguous timeout"), _Response()])

        async def run_call():
            return await env._call("step", params={"action_str": "x"})

        with self.assertRaisesRegex(RuntimeError, "Remote call failed"):
            asyncio.run(run_call())
        self.assertEqual(env._client.urls, ["http://server-a/call"])

    def test_session_bound_retries_stay_on_connecting_server(self) -> None:
        try:
            from vagen.envs_remote import gym_image_env_client as module
        except ImportError as exc:
            self.skipTest(f"remote-client dependencies unavailable: {exc}")

        env = module.GymImageEnvClient(
            {
                "base_urls": ["http://server-a", "http://server-b"],
                "retries": 1,
                "backoff": 0.0,
                "max_delay": 0.0,
                "log_retries": False,
            }
        )
        env._session_id = "session-on-a"
        env._current_url_index = 0
        env._client = _Client([RuntimeError("transient"), _Response()])

        async def run_call():
            with (
                mock.patch.object(
                    module,
                    "decode_multipart",
                    return_value=({"obs": "ok"}, None),
                ),
                mock.patch.object(module.asyncio, "sleep", return_value=None),
            ):
                return await env._call("system_prompt")

        data, images = asyncio.run(run_call())
        self.assertEqual(data, {"obs": "ok"})
        self.assertIsNone(images)
        self.assertEqual(
            env._client.urls,
            ["http://server-a/call", "http://server-a/call"],
        )


if __name__ == "__main__":
    unittest.main()
