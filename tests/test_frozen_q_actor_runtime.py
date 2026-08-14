from __future__ import annotations

import unittest


class FrozenQActorRuntimeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        import ray

        cls._owns_ray = not ray.is_initialized()
        if cls._owns_ray:
            ray.init(
                num_cpus=2,
                include_dashboard=False,
                log_to_driver=False,
            )

    @classmethod
    def tearDownClass(cls) -> None:
        import ray

        if cls._owns_ray:
            ray.shutdown()

    def _snapshot_state(self, source_step: int, *, mutate: bool = False):
        import torch

        from nimloth.training.rl.joint_critic import (
            JointActionValueCritic,
            create_frozen_critic_snapshot,
            export_frozen_critic_snapshot,
        )
        from nimloth.wm.grid import SharedSlotProjector
        from nimloth.wm.value_head import ValueHead

        critic = JointActionValueCritic(
            state_projector=SharedSlotProjector(
                input_dim=3,
                output_dim=2,
                hidden_dim=5,
                grid_tokens=2,
            ),
            value_head=ValueHead(emb_dim=2, num_actions=3, hidden_dim=4),
        )
        if mutate:
            with torch.no_grad():
                next(critic.parameters()).add_(0.25)
        return export_frozen_critic_snapshot(
            create_frozen_critic_snapshot(
                critic,
                source_step=source_step,
                contract_id="sha256:joint-contract",
                score_dtype="float32",
            )
        )

    def test_real_actor_is_cpu_only_and_preserves_lifecycle(self) -> None:
        import ray

        from vagen.joint_policy.frozen_q_actor import FrozenQScoringActor

        initial = self._snapshot_state(776)
        actor = FrozenQScoringActor.remote(
            initial.to_mapping(),
            activation_version=4,
        )
        try:
            status = ray.get(actor.status.remote())
            self.assertEqual(status["active_snapshot_id"], initial.snapshot_id)
            self.assertEqual(status["active_source_step"], 776)
            self.assertEqual(status["activation_version"], 4)
            self.assertEqual(status["torch_num_threads"], 1)
            self.assertEqual(status["torch_num_interop_threads"], 1)

            pin = ray.get(
                actor.pin_batch.remote(
                    {
                        "batch_id": "batch-1",
                        "policy_step": 1,
                        "expected_snapshot_id": initial.snapshot_id,
                        "expected_activation_version": 4,
                    }
                )
            )
            candidate = self._snapshot_state(777, mutate=True)
            ray.get(
                actor.stage_snapshot.remote(
                    {
                        "new_snapshot_state": candidate.to_mapping(),
                        "expected_active_snapshot_id": initial.snapshot_id,
                        "expected_activation_version": 4,
                    }
                )
            )
            with self.assertRaisesRegex(Exception, "open batch"):
                ray.get(
                    actor.activate_staged.remote(
                        {
                            "staged_snapshot_id": candidate.snapshot_id,
                            "expected_active_snapshot_id": initial.snapshot_id,
                            "expected_activation_version": 4,
                        }
                    )
                )
            ray.get(actor.unpin_batch.remote(pin))
            activated = ray.get(
                actor.activate_staged.remote(
                    {
                        "staged_snapshot_id": candidate.snapshot_id,
                        "expected_active_snapshot_id": initial.snapshot_id,
                        "expected_activation_version": 4,
                    }
                )
            )
            self.assertEqual(activated["active_snapshot_id"], candidate.snapshot_id)
            self.assertEqual(activated["activation_version"], 5)
            checkpoint = ray.get(actor.checkpoint_state.remote())
            self.assertEqual(
                checkpoint["active_snapshot_state"]["snapshot_id"],
                candidate.snapshot_id,
            )
        finally:
            ray.kill(actor, no_restart=True)

    def test_real_actor_rejects_malformed_snapshot_during_construction(self) -> None:
        import ray

        from vagen.joint_policy.frozen_q_actor import FrozenQScoringActor

        malformed = self._snapshot_state(776).to_mapping()
        malformed["snapshot_id"] = "sha256:forged"
        actor = FrozenQScoringActor.remote(malformed)
        try:
            with self.assertRaisesRegex(Exception, "fingerprint|snapshot_id"):
                ray.get(actor.status.remote())
        finally:
            try:
                ray.kill(actor, no_restart=True)
            except Exception:
                pass


if __name__ == "__main__":
    unittest.main()
