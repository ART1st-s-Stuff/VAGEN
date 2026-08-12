from __future__ import annotations

import argparse
import json
import math
import tempfile
import unittest
from pathlib import Path


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
            env_timeout=500.0,
        )

    def test_config_has_no_optimizer_actor_critic_or_fsdp(self) -> None:
        try:
            from vagen.standalone_one_turn_smoke import build_config
        except ImportError as exc:
            self.skipTest(f"smoke dependencies unavailable: {exc}")

        config = build_config(self._args(Path("/output.json")))
        text = str(config)
        self.assertEqual(config.actor_rollout_ref.rollout.name, "nimloth_vllm")
        self.assertEqual(config.actor_rollout_ref.rollout.agent.num_workers, 1)
        self.assertTrue(config.decision_ledger.enabled)
        self.assertFalse(config.joint_policy.enabled)
        for forbidden in ("optimizer", "fsdp", "critic"):
            self.assertNotIn(forbidden, text.lower())

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
