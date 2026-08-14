import unittest


class _Manager:
    def __init__(self, status):
        self.status = dict(status)
        self.calls = []

    def frozen_q_status(self):
        self.calls.append(("status", None))
        return dict(self.status)

    def stage_frozen_q_snapshot(self, request):
        self.calls.append(("stage", request))
        self.status["staged_snapshot_id"] = request["new_snapshot_state"]["snapshot_id"]
        return dict(self.status)

    def activate_staged_frozen_q_snapshot(self, request):
        self.calls.append(("activate", request))
        self.status.update(
            {
                "active_snapshot_id": request["staged_snapshot_id"],
                "active_source_step": 777,
                "activation_version": self.status["activation_version"] + 1,
                "staged_snapshot_id": None,
            }
        )
        return dict(self.status)


class JointUpdateTransactionTest(unittest.TestCase):
    def _exports(self):
        state = {
            "schema": "nimloth_frozen_critic_snapshot_state_v1",
            "source_step": 777,
            "contract_id": "contract-1",
            "snapshot_id": "snapshot-new",
            "score_dtype": "float32",
            "critic_spec": {},
            "critic_state": {},
        }
        return [
            {
                "rank": rank,
                "world_size": 2,
                "completed_updates": 1,
                "source_step": 777,
                "snapshot_id": "snapshot-new",
                "contract_id": "contract-1",
                "score_dtype": "float32",
                "optimizer_fingerprint": "sha256:optimizer",
                "snapshot_state": state if rank == 0 else None,
            }
            for rank in range(2)
        ]

    def test_validates_all_ranks_then_stages_and_activates(self) -> None:
        from vagen.joint_policy.update_transaction import (
            publish_replicated_joint_snapshot,
        )

        manager = _Manager(
            {
                "active_snapshot_id": "snapshot-old",
                "active_source_step": 776,
                "contract_id": "contract-1",
                "score_dtype": "float32",
                "activation_version": 3,
                "staged_snapshot_id": None,
                "open_batch_count": 0,
            }
        )
        result = publish_replicated_joint_snapshot(
            manager=manager,
            rank_exports=self._exports(),
            expected_world_size=2,
            expected_active_snapshot_id="snapshot-old",
            expected_active_source_step=776,
            expected_activation_version=3,
        )
        self.assertEqual(result["active_snapshot_id"], "snapshot-new")
        self.assertEqual(result["active_source_step"], 777)
        self.assertEqual([name for name, _ in manager.calls], ["status", "stage", "activate"])

    def test_rejects_rank_or_optimizer_divergence_before_stage(self) -> None:
        from vagen.joint_policy.update_transaction import (
            publish_replicated_joint_snapshot,
        )

        for mutation, pattern in (
            ((1, "rank", 0), "ranks"),
            ((1, "snapshot_id", "other"), "snapshot"),
            ((1, "optimizer_fingerprint", "other"), "optimizer"),
        ):
            exports = self._exports()
            row, field, value = mutation
            exports[row][field] = value
            manager = _Manager(
                {
                    "active_snapshot_id": "snapshot-old",
                    "active_source_step": 776,
                    "contract_id": "contract-1",
                    "score_dtype": "float32",
                    "activation_version": 3,
                    "staged_snapshot_id": None,
                    "open_batch_count": 0,
                }
            )
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, pattern):
                publish_replicated_joint_snapshot(
                    manager=manager,
                    rank_exports=exports,
                    expected_world_size=2,
                    expected_active_snapshot_id="snapshot-old",
                    expected_active_source_step=776,
                    expected_activation_version=3,
                )
            self.assertEqual(manager.calls, [])

    def test_rejects_open_pin_or_staged_candidate_before_mutation(self) -> None:
        from vagen.joint_policy.update_transaction import (
            publish_replicated_joint_snapshot,
        )

        for field, value, pattern in (
            ("open_batch_count", 1, "open"),
            ("staged_snapshot_id", "stale", "staged"),
        ):
            status = {
                "active_snapshot_id": "snapshot-old",
                "active_source_step": 776,
                "contract_id": "contract-1",
                "score_dtype": "float32",
                "activation_version": 3,
                "staged_snapshot_id": None,
                "open_batch_count": 0,
            }
            status[field] = value
            manager = _Manager(status)
            with self.assertRaisesRegex(ValueError, pattern):
                publish_replicated_joint_snapshot(
                    manager=manager,
                    rank_exports=self._exports(),
                    expected_world_size=2,
                    expected_active_snapshot_id="snapshot-old",
                    expected_active_source_step=776,
                    expected_activation_version=3,
                )
            self.assertEqual([name for name, _ in manager.calls], ["status"])


if __name__ == "__main__":
    unittest.main()
