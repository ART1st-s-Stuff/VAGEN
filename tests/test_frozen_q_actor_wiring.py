from __future__ import annotations

import ast
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
MANAGER = ROOT / "vagen" / "agent_loop" / "agent_loop_no_concat.py"


class FrozenQActorWiringTest(unittest.TestCase):
    def test_manager_creates_one_cpu_owner_before_agent_workers(self) -> None:
        source = MANAGER.read_text(encoding="utf-8")
        tree = ast.parse(source)
        manager = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "AgentLoopManager"
        )
        init = next(
            node
            for node in manager.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        calls = [
            node.func.attr
            for node in ast.walk(init)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        self.assertIn("_init_frozen_q_owner", calls)
        self.assertLess(
            calls.index("_init_frozen_q_owner"),
            calls.index("_init_agent_loop_workers"),
        )
        self.assertIn("initial_frozen_q_snapshot_state", source)

    def test_actor_is_exactly_one_cpu_and_zero_gpu(self) -> None:
        actor_path = ROOT / "vagen" / "joint_policy" / "frozen_q_actor.py"
        source = actor_path.read_text(encoding="utf-8")
        self.assertIn("num_cpus=1", source)
        self.assertIn("num_gpus=0", source)
        self.assertNotIn("checkpoint_root", source)
        self.assertNotIn("torch.optim", source)

    def test_manager_passes_same_owner_handle_to_every_worker(self) -> None:
        source = MANAGER.read_text(encoding="utf-8")
        self.assertIn("self.frozen_q_owner", source)
        self.assertIn("if self.frozen_q_owner is not None", source)
        self.assertIn("(self.frozen_q_owner,)", source)
        worker_init = source[source.index("class AgentLoopWorker"):source.index("async def get_trajectory_info")]
        self.assertIn("frozen_q_owner", worker_init)

    def test_disabled_mode_preserves_legacy_constructor_calls(self) -> None:
        source = MANAGER.read_text(encoding="utf-8")
        run_start = source.index("async def _run_agent_loop")
        run_end = source.index("def _postprocess", run_start)
        run_method = source[run_start:run_end]
        self.assertIn('if self.frozen_q_owner is not None:', run_method)
        self.assertIn('instantiate_kwargs["frozen_q_owner"]', run_method)
        self.assertNotIn("frozen_q_owner=self.frozen_q_owner", run_method)

        workers_start = source.index("def _init_agent_loop_workers")
        workers_end = source.index("def generate_sequences", workers_start)
        workers_method = source[workers_start:workers_end]
        self.assertIn("if self.frozen_q_owner is not None", workers_method)
        self.assertIn("else ()", workers_method)

    def test_disabled_custom_worker_receives_only_legacy_arguments(self) -> None:
        try:
            from vagen.agent_loop.agent_loop_no_concat import AgentLoopManager
        except ImportError as exc:
            self.skipTest(f"manager dependencies unavailable: {exc}")

        calls = []

        class _WorkerClass:
            @classmethod
            def options(cls, **_kwargs):
                return cls()

            def remote(self, *args):
                calls.append(args)
                return object()

        manager = object.__new__(AgentLoopManager)
        manager.config = SimpleNamespace(
            actor_rollout_ref=SimpleNamespace(
                rollout=SimpleNamespace(
                    agent=SimpleNamespace(num_workers=1)
                )
            )
        )
        manager.agent_loop_workers_class = _WorkerClass
        manager.server_handles = ["server"]
        manager.reward_router_address = "reward"
        manager.frozen_q_owner = None
        with patch(
            "vagen.agent_loop.agent_loop_no_concat.ray.nodes",
            return_value=[
                {
                    "NodeID": "01" * 28,
                    "Alive": True,
                    "Resources": {"CPU": 1},
                }
            ],
        ):
            manager._init_agent_loop_workers()
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0], (manager.config, ["server"], "reward"))


if __name__ == "__main__":
    unittest.main()
