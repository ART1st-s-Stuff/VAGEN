from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf


def _nested_mapping_keys(value):
    if isinstance(value, dict):
        for key, nested in value.items():
            yield str(key).lower()
            yield from _nested_mapping_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _nested_mapping_keys(nested)


class StandaloneOneTurnSmokeTest(unittest.TestCase):
    def _args(self, output: Path) -> argparse.Namespace:
        root = Path(__file__).resolve().parents[1]
        return argparse.Namespace(
            model=Path("/model"),
            env_url="http://127.0.0.1:8000",
            output=output,
            run_name="1_smoke_vagen_k16_one_turn",
            agent_loop_config=root / "vagen/configs/agent_no_concat.yaml",
            eval_set="base",
            seed=0,
            latent_token_count=16,
            prompt_length=9000,
            response_length=512,
            temperature=0.0,
            top_p=1.0,
            gpu_memory_utilization=0.6,
            tensor_parallel_size=8,
            env_timeout=300.0,
        )

    def test_config_has_no_optimizer_actor_critic_or_fsdp(self) -> None:
        try:
            from vagen.standalone_one_turn_smoke import build_config
        except ImportError as exc:
            self.skipTest(f"smoke dependencies unavailable: {exc}")

        config = build_config(self._args(Path("/output.json")))
        config_keys = set(
            _nested_mapping_keys(OmegaConf.to_container(config, resolve=True))
        )
        self.assertEqual(config.actor_rollout_ref.rollout.name, "nimloth_vllm")
        self.assertEqual(
            config.actor_rollout_ref.rollout.tensor_model_parallel_size,
            8,
        )
        self.assertEqual(config.trainer.n_gpus_per_node, 8)
        self.assertEqual(
            config.actor_rollout_ref.rollout.engine_kwargs.vllm.mm_encoder_tp_mode,
            "data",
        )
        self.assertEqual(config.actor_rollout_ref.rollout.agent.num_workers, 1)
        self.assertTrue(config.decision_ledger.enabled)
        self.assertFalse(config.joint_policy.enabled)
        for forbidden in ("optimizer", "fsdp", "critic"):
            self.assertFalse(
                any(forbidden in key for key in config_keys),
                f"unexpected {forbidden} config key",
            )

    def test_input_config_matches_current_navigation_profile(self) -> None:
        try:
            from vagen.standalone_one_turn_smoke import build_input
        except ImportError as exc:
            self.skipTest(f"smoke dependencies unavailable: {exc}")

        config = build_input(self._args(Path("/output.json"))).non_tensor_batch[
            "config"
        ][0]
        self.assertEqual(
            config,
            {
                "base_urls": "http://127.0.0.1:8000",
                "timeout": 300.0,
                "retries": 0,
                "eval_set": "base",
                "prompt_format": "nimloth",
                "latent_token_count": 16,
                "max_actions_per_step": 1,
                "action_sep": "|",
                "example_count": 0,
                "format_reward": 0.0,
                "per_turn_format_reward": 0.0,
                "success_reward": 10.0,
                "success_threshold": 1.5,
                "step_length": 0.5,
            },
        )

    def test_input_seed_is_json_serializable_python_int(self) -> None:
        try:
            from vagen.envs_remote.multipart_codec import encode_multipart
            from vagen.standalone_one_turn_smoke import build_input
        except ImportError as exc:
            self.skipTest(f"smoke dependencies unavailable: {exc}")

        row = build_input(self._args(Path("/output.json")))
        seed = row.non_tensor_batch["seed"][0]
        self.assertIs(type(seed), int)
        _boundary, body = encode_multipart(
            {"seed": seed, "config": row.non_tensor_batch["config"][0]}
        )
        self.assertIn(b'"seed": 0', body)

    def test_custom_rollout_module_imports_and_registers(self) -> None:
        try:
            from verl.workers.rollout.replica import RolloutReplicaRegistry
            import vagen.rollout.nimloth_vllm  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"smoke dependencies unavailable: {exc}")

        replica = RolloutReplicaRegistry.get("nimloth_vllm")
        self.assertEqual(replica.__name__, "NimlothVLLMReplica")

    def test_ray_runtime_env_preserves_explicit_runtime_paths(self) -> None:
        try:
            from unittest.mock import patch
            from vagen.standalone_one_turn_smoke import build_ray_runtime_env
        except ImportError as exc:
            self.skipTest(f"smoke dependencies unavailable: {exc}")

        with patch.dict(
            "os.environ",
            {
                "PATH": "/venv/bin:/usr/bin",
                "PYTHONPATH": "/nimloth/src:/vagen:/verl",
                "RAY_TMPDIR": "/tmp/does-not-propagate-as-env",
            },
            clear=True,
        ):
            actor_env = build_ray_runtime_env()["env_vars"]
        self.assertEqual(actor_env["PATH"], "/venv/bin:/usr/bin")
        self.assertEqual(actor_env["PYTHONPATH"], "/nimloth/src:/vagen:/verl")
        self.assertEqual(actor_env["VLLM_USE_V1"], "1")
        self.assertNotIn("RAY_TMPDIR", actor_env)

    def test_atomic_json_rejects_nonfinite_values(self) -> None:
        try:
            from vagen.standalone_one_turn_smoke import atomic_write_json
        except ImportError as exc:
            self.skipTest(f"smoke dependencies unavailable: {exc}")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            with self.assertRaises(ValueError):
                atomic_write_json(path, {"bad": math.inf})
            self.assertFalse(path.exists())

    def test_atomic_json_round_trip(self) -> None:
        try:
            from vagen.standalone_one_turn_smoke import atomic_write_json
        except ImportError as exc:
            self.skipTest(f"smoke dependencies unavailable: {exc}")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "result.json"
            atomic_write_json(path, {"status": "passed", "optimizer": None})
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")),
                {"status": "passed", "optimizer": None},
            )


if __name__ == "__main__":
    unittest.main()
