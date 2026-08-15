import hashlib
import tempfile
import unittest
from pathlib import Path


_DATA = b"dataloader-state"
_DATA_DIGEST = f"sha256:{hashlib.sha256(_DATA).hexdigest()}"
_TRAINING_CONTRACT_ID = "sha256:" + "1" * 64


class JointPolicyCheckpointTest(unittest.TestCase):
    def setUp(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            self.skipTest(f"torch unavailable: {exc}")

    def _exports(self):
        import torch

        actor = {
            "schema": "vagen_joint_actor_critic_checkpoint_v1",
            "completed_updates": 1,
            "source_step": 777,
            "snapshot_id": "snapshot-777",
            "contract_id": "contract-1",
            "score_dtype": "float32",
            "critic_state": {"weight": torch.tensor([1.0])},
            "critic_optimizer_state": {"state": {}, "param_groups": []},
            "critic_optimizer_fingerprint": "sha256:optimizer",
        }
        return [
            {
                "rank": rank,
                "world_size": 2,
                "completed_updates": 1,
                "source_step": 777,
                "snapshot_id": "snapshot-777",
                "contract_id": "contract-1",
                "score_dtype": "float32",
                "optimizer_fingerprint": "sha256:optimizer",
                "checkpoint_payload": actor if rank == 0 else None,
            }
            for rank in range(2)
        ]

    def _owner(self):
        return {
            "schema": "nimloth_frozen_q_owner_checkpoint_v1",
            "activation_version": 1,
            "active_snapshot_state": {
                "source_step": 777,
                "snapshot_id": "snapshot-777",
                "contract_id": "contract-1",
                "score_dtype": "float32",
            },
        }

    def test_assembles_k4_planner_payload_and_transport_owner(self) -> None:
        from vagen.joint_policy.checkpoint import assemble_joint_checkpoint

        exports = self._exports()
        actor = dict(exports[0]["checkpoint_payload"])
        actor.pop("critic_state")
        actor.pop("critic_optimizer_state")
        actor.pop("critic_optimizer_fingerprint")
        actor.update(
            {
                "schema": "vagen_joint_k4_actor_planning_checkpoint_v1",
                "planning_state": {},
                "planning_optimizer_state": {"state": {}, "param_groups": []},
                "planning_optimizer_fingerprint": "sha256:optimizer",
                "snapshot_transport": {},
            }
        )
        exports[0]["checkpoint_payload"] = actor
        owner = self._owner()
        owner["active_snapshot_state"] = {
            "schema": "vagen_frozen_k4_planner_transport_v1",
            "transport_path": "/shared/frozen_k4_planner.pt",
            "snapshot_source_step": 777,
            "snapshot_id": "snapshot-777",
            "contract_id": "contract-1",
            "score_dtype": "float32",
            "planning_horizon": 4,
            "mcts_num_simulations": 100,
            "mcts_exploration_constant": 1.0,
        }
        payload = assemble_joint_checkpoint(
            global_step=1,
            run_seed=42,
            rank_exports=exports,
            owner_checkpoint_state=owner,
            expected_world_size=2,
            dataloader_sha256=_DATA_DIGEST,
            training_contract_id=_TRAINING_CONTRACT_ID,
        )
        self.assertEqual(payload["actor_critic"]["planning_state"], {})

    def test_atomic_marker_round_trip_and_latest_complete_selection(self) -> None:
        from vagen.joint_policy.checkpoint import (
            assemble_joint_checkpoint,
            find_latest_complete_joint_checkpoint,
            load_complete_joint_checkpoint,
            save_atomic_joint_checkpoint,
        )

        payload = assemble_joint_checkpoint(
            global_step=1,
            run_seed=42,
            rank_exports=self._exports(),
            owner_checkpoint_state=self._owner(),
            expected_world_size=2,
            dataloader_sha256=_DATA_DIGEST,
            training_contract_id=_TRAINING_CONTRACT_ID,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            complete = root / "global_step_1"
            incomplete = root / "global_step_2"
            incomplete.mkdir()
            complete.mkdir()
            (complete / "data.pt").write_bytes(_DATA)
            save_atomic_joint_checkpoint(complete, payload)
            self.assertEqual(
                find_latest_complete_joint_checkpoint(root),
                str(complete),
            )
            loaded = load_complete_joint_checkpoint(complete)
            self.assertEqual(loaded["global_step"], 1)
            self.assertEqual(
                loaded["actor_critic"]["snapshot_id"],
                "snapshot-777",
            )

    def test_sidecar_corruption_fails_closed(self) -> None:
        from vagen.joint_policy.checkpoint import (
            assemble_joint_checkpoint,
            load_complete_joint_checkpoint,
            save_atomic_joint_checkpoint,
        )

        payload = assemble_joint_checkpoint(
            global_step=1,
            run_seed=42,
            rank_exports=self._exports(),
            owner_checkpoint_state=self._owner(),
            expected_world_size=2,
            dataloader_sha256=_DATA_DIGEST,
            training_contract_id=_TRAINING_CONTRACT_ID,
        )
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "global_step_1"
            folder.mkdir()
            (folder / "data.pt").write_bytes(_DATA)
            save_atomic_joint_checkpoint(folder, payload)
            with (folder / "joint_training.pt").open("ab") as handle:
                handle.write(b"corrupt")
            with self.assertRaisesRegex(ValueError, "digest"):
                load_complete_joint_checkpoint(folder)

    def test_rejects_divergent_rank_before_assembly(self) -> None:
        from vagen.joint_policy.checkpoint import assemble_joint_checkpoint

        exports = self._exports()
        exports[1]["optimizer_fingerprint"] = "sha256:different"
        with self.assertRaisesRegex(ValueError, "optimizer"):
            assemble_joint_checkpoint(
                global_step=1,
                run_seed=42,
                rank_exports=exports,
                owner_checkpoint_state=self._owner(),
                expected_world_size=2,
                dataloader_sha256=_DATA_DIGEST,
                training_contract_id=_TRAINING_CONTRACT_ID,
            )


if __name__ == "__main__":
    unittest.main()
